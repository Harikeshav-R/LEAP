#ifndef LEAP_TRANSFORMER_H
#define LEAP_TRANSFORMER_H

#include "Config.h"
#include "Transport.h"
#include <string>
#include <memory>
#include <vector>

namespace Inference {
    enum class DistributedMode {
        Single,
        Master,
        Worker
    };

    struct DistributedConfig {
        DistributedMode mode = DistributedMode::Single;
        int split_layer = 0;
        int end_layer = 0; // The layer this node stops at (exclusive)
        Transport *transport = nullptr;
        bool is_tail = false;
    };

    struct PacketHeader {
        int pos;
        int flags; // 0: No Reply (Fire-and-Forget), 1: Need Reply
    };

    constexpr int FLAG_NO_REPLY = 0;
    constexpr int FLAG_NEED_REPLY = 1;

    class Transformer {
    public:
        Config config{};
        DistributedConfig dist_config{};

        virtual ~Transformer() = default;

        // The core function: forward pass
        virtual float *forward(int token, int pos, int flags = FLAG_NEED_REPLY) = 0;

        // Worker loop: receive tensor, process layers, send back
        virtual void worker_loop() = 0;

        void set_distributed_config(const DistributedConfig &config) {
            dist_config = config;
        }

        // Update layer configuration at runtime (for dynamic resizing)
        virtual void update_layer_config(const ControlMessage &msg) {
            dist_config.split_layer = msg.split_layer;
            dist_config.end_layer = msg.end_layer;
            dist_config.is_tail = msg.is_tail;
        }

        // Access KV cache data (for KV transfer protocol)
        virtual float* get_key_cache() = 0;
        virtual float* get_value_cache() = 0;

        // Clear KV cache (needed after layer redistribution to avoid stale state)
        virtual void clear_kv_cache() = 0;

        // KV Cache Transfer: Master initiates the ring transfer after resize
        // Sends departing layer KV data forward, then receives returning layers from tail
        void initiate_kv_transfer(float *key_cache, float *value_cache,
                                  int pos, int kv_dim, int seq_len,
                                  int old_split, int new_split) {
            if (!dist_config.transport || pos <= 0) return;

            const size_t slice_data_size = static_cast<size_t>(pos) * kv_dim * sizeof(float);
            const size_t packet_size = dist_config.transport->get_packet_size();

            // Build bundle: layers master is giving up (old range that's no longer ours)
            // Master owns [0, split_layer). Old: [0, old_split), New: [0, new_split)
            std::vector<int32_t> departing_layers;
            for (int l = new_split; l < old_split; l++) {
                departing_layers.push_back(l);  // Master lost these layers
            }

            // Send KV transfer header (padded to packet_size)
            KvTransferHeader hdr{};
            hdr.magic = KV_TRANSFER_MAGIC;
            hdr.num_slices = static_cast<int32_t>(departing_layers.size());
            hdr.pos = pos;
            hdr.kv_dim = kv_dim;

            std::vector<char> hdr_buf(packet_size, 0);
            std::memcpy(hdr_buf.data(), &hdr, sizeof(hdr));
            dist_config.transport->send_next(hdr_buf.data(), packet_size);

            // Send each departing layer's KV data
            for (int32_t layer_id : departing_layers) {
                // Send layer_id
                dist_config.transport->send_next(&layer_id, sizeof(layer_id));
                // Send key cache for this layer
                float *key_ptr = key_cache + static_cast<size_t>(layer_id) * seq_len * kv_dim;
                dist_config.transport->send_next(key_ptr, slice_data_size);
                // Send value cache for this layer
                float *val_ptr = value_cache + static_cast<size_t>(layer_id) * seq_len * kv_dim;
                dist_config.transport->send_next(val_ptr, slice_data_size);
            }

            // Now receive the return bundle (layers coming back from workers)
            std::vector<char> recv_hdr_buf(packet_size);
            dist_config.transport->recv_prev(recv_hdr_buf.data(), packet_size);

            KvTransferHeader recv_hdr{};
            std::memcpy(&recv_hdr, recv_hdr_buf.data(), sizeof(recv_hdr));

            if (recv_hdr.magic == KV_TRANSFER_MAGIC && recv_hdr.num_slices > 0) {
                for (int i = 0; i < recv_hdr.num_slices; i++) {
                    int32_t layer_id;
                    dist_config.transport->recv_prev(&layer_id, sizeof(layer_id));

                    // Check if this layer is in master's new range [0, new_split)
                    if (layer_id >= 0 && layer_id < new_split) {
                        float *key_ptr = key_cache + static_cast<size_t>(layer_id) * seq_len * kv_dim;
                        dist_config.transport->recv_prev(key_ptr, slice_data_size);
                        float *val_ptr = value_cache + static_cast<size_t>(layer_id) * seq_len * kv_dim;
                        dist_config.transport->recv_prev(val_ptr, slice_data_size);
                    } else {
                        // Discard (shouldn't happen after full ring rotation)
                        std::vector<char> discard(slice_data_size);
                        dist_config.transport->recv_prev(discard.data(), slice_data_size);
                        dist_config.transport->recv_prev(discard.data(), slice_data_size);
                    }
                }
            }
        }

        // KV Cache Transfer: Worker handles an incoming KV bundle
        // Extracts layers it needs, adds its departing layers, forwards the rest
        void handle_kv_transfer(float *key_cache, float *value_cache,
                                int kv_dim, int seq_len,
                                int old_start, int old_end,
                                int new_start, int new_end,
                                const KvTransferHeader &incoming_hdr) {
            const int pos = incoming_hdr.pos;
            const size_t slice_data_size = static_cast<size_t>(pos) * kv_dim * sizeof(float);
            const size_t packet_size = dist_config.transport->get_packet_size();

            // Receive all incoming slices
            struct KvSlice {
                int32_t layer_id;
                std::vector<float> key_data;
                std::vector<float> value_data;
            };
            std::vector<KvSlice> incoming_slices;

            for (int i = 0; i < incoming_hdr.num_slices; i++) {
                KvSlice slice;
                dist_config.transport->recv_prev(&slice.layer_id, sizeof(slice.layer_id));
                slice.key_data.resize(static_cast<size_t>(pos) * kv_dim);
                slice.value_data.resize(static_cast<size_t>(pos) * kv_dim);
                dist_config.transport->recv_prev(slice.key_data.data(), slice_data_size);
                dist_config.transport->recv_prev(slice.value_data.data(), slice_data_size);
                incoming_slices.push_back(std::move(slice));
            }

            // Extract layers this worker needs (in new range but NOT in old range)
            for (auto &slice : incoming_slices) {
                if (slice.layer_id >= new_start && slice.layer_id < new_end) {
                    // Copy into our KV cache
                    float *key_ptr = key_cache + static_cast<size_t>(slice.layer_id) * seq_len * kv_dim;
                    float *val_ptr = value_cache + static_cast<size_t>(slice.layer_id) * seq_len * kv_dim;
                    std::memcpy(key_ptr, slice.key_data.data(), slice_data_size);
                    std::memcpy(val_ptr, slice.value_data.data(), slice_data_size);
                    slice.layer_id = -1;  // Mark as consumed
                }
            }

            // Build forward bundle: unconsumed incoming slices + our departing layers
            std::vector<KvSlice> forward_slices;

            // Add unconsumed incoming slices
            for (auto &slice : incoming_slices) {
                if (slice.layer_id >= 0) {
                    forward_slices.push_back(std::move(slice));
                }
            }

            // Add our departing layers (in old range but NOT in new range)
            for (int l = old_start; l < old_end; l++) {
                if (l < new_start || l >= new_end) {
                    KvSlice slice;
                    slice.layer_id = l;
                    slice.key_data.resize(static_cast<size_t>(pos) * kv_dim);
                    slice.value_data.resize(static_cast<size_t>(pos) * kv_dim);
                    float *key_ptr = key_cache + static_cast<size_t>(l) * seq_len * kv_dim;
                    float *val_ptr = value_cache + static_cast<size_t>(l) * seq_len * kv_dim;
                    std::memcpy(slice.key_data.data(), key_ptr, slice_data_size);
                    std::memcpy(slice.value_data.data(), val_ptr, slice_data_size);
                    forward_slices.push_back(std::move(slice));
                }
            }

            // Send forward bundle header
            KvTransferHeader fwd_hdr{};
            fwd_hdr.magic = KV_TRANSFER_MAGIC;
            fwd_hdr.num_slices = static_cast<int32_t>(forward_slices.size());
            fwd_hdr.pos = pos;
            fwd_hdr.kv_dim = kv_dim;

            std::vector<char> hdr_buf(packet_size, 0);
            std::memcpy(hdr_buf.data(), &fwd_hdr, sizeof(fwd_hdr));
            dist_config.transport->send_next(hdr_buf.data(), packet_size);

            // Send each forwarded slice
            for (auto &slice : forward_slices) {
                dist_config.transport->send_next(&slice.layer_id, sizeof(slice.layer_id));
                dist_config.transport->send_next(slice.key_data.data(), slice_data_size);
                dist_config.transport->send_next(slice.value_data.data(), slice_data_size);
            }
        }

        // Factory method to create the appropriate Transformer (Float or Quantized) based on file
        static std::unique_ptr<Transformer> create(const std::string &checkpoint_path);

    protected:
        // Optimization: Reusable buffer for network transfers to avoid repeated allocations
        std::vector<char> transfer_buffer;
    };
} // namespace Inference


#endif // LEAP_TRANSFORMER_H