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

        // KV Cache Transfer: Two-Phase Protocol
        // Phase 1: Header (padded to packet_size) — detected inline in worker_loop
        // Phase 2: Payload (contiguous: [layer_id + key + value] per slice)
        // This ensures compatibility with all transports (TCP, UDP, Kernel).

        // Master initiates the ring transfer after resize
        void initiate_kv_transfer(float *key_cache, float *value_cache,
                                  int pos, int kv_dim, int seq_len,
                                  int old_split, int new_split) {
            if (!dist_config.transport || pos <= 0) return;

            const size_t slice_bytes = static_cast<size_t>(pos) * kv_dim * sizeof(float);
            const size_t per_slice_size = sizeof(int32_t) + slice_bytes * 2;
            const size_t packet_size = dist_config.transport->get_packet_size();

            // Layers master is giving up: [new_split, old_split)
            std::vector<int32_t> departing_layers;
            for (int l = new_split; l < old_split; l++) {
                departing_layers.push_back(l);
            }

            // Phase 1: Send header padded to packet_size
            KvTransferHeader hdr{};
            hdr.magic = KV_TRANSFER_MAGIC;
            hdr.num_slices = static_cast<int32_t>(departing_layers.size());
            hdr.pos = pos;
            hdr.kv_dim = kv_dim;

            std::vector<char> hdr_buf(packet_size, 0);
            std::memcpy(hdr_buf.data(), &hdr, sizeof(hdr));
            dist_config.transport->send_next(hdr_buf.data(), packet_size);

            // Phase 2: Build and send payload
            const size_t payload_size = departing_layers.size() * per_slice_size;
            std::vector<char> payload(payload_size);
            char *wp = payload.data();

            for (int32_t layer_id : departing_layers) {
                std::memcpy(wp, &layer_id, sizeof(layer_id));
                wp += sizeof(layer_id);
                float *kp = key_cache + static_cast<size_t>(layer_id) * seq_len * kv_dim;
                std::memcpy(wp, kp, slice_bytes);
                wp += slice_bytes;
                float *vp = value_cache + static_cast<size_t>(layer_id) * seq_len * kv_dim;
                std::memcpy(wp, vp, slice_bytes);
                wp += slice_bytes;
            }

            if (payload_size > 0) {
                dist_config.transport->send_next(payload.data(), payload_size);
            }

            // Receive return: Phase 1 (header in packet_size)
            std::vector<char> recv_hdr_buf(packet_size);
            dist_config.transport->recv_prev(recv_hdr_buf.data(), packet_size);

            KvTransferHeader recv_hdr{};
            std::memcpy(&recv_hdr, recv_hdr_buf.data(), sizeof(recv_hdr));

            if (recv_hdr.magic == KV_TRANSFER_MAGIC && recv_hdr.num_slices > 0) {
                // Receive return: Phase 2 (payload)
                const size_t recv_payload_size = recv_hdr.num_slices * per_slice_size;
                std::vector<char> recv_payload(recv_payload_size);
                dist_config.transport->recv_prev(recv_payload.data(), recv_payload_size);

                const char *rp = recv_payload.data();
                for (int i = 0; i < recv_hdr.num_slices; i++) {
                    int32_t layer_id;
                    std::memcpy(&layer_id, rp, sizeof(layer_id));
                    rp += sizeof(layer_id);

                    if (layer_id >= 0 && layer_id < new_split) {
                        float *kp = key_cache + static_cast<size_t>(layer_id) * seq_len * kv_dim;
                        std::memcpy(kp, rp, slice_bytes);
                        rp += slice_bytes;
                        float *vp = value_cache + static_cast<size_t>(layer_id) * seq_len * kv_dim;
                        std::memcpy(vp, rp, slice_bytes);
                        rp += slice_bytes;
                    } else {
                        rp += slice_bytes * 2;
                    }
                }
            }
        }

        // Worker handles KV transfer: called after detecting KV_TRANSFER_MAGIC in transfer_buffer
        // The header has already been received in the worker_loop's recv_prev.
        void handle_kv_transfer(float *key_cache, float *value_cache,
                                int kv_dim, int seq_len,
                                int old_start, int old_end,
                                int new_start, int new_end,
                                const KvTransferHeader &incoming_hdr) {
            const int pos = incoming_hdr.pos;
            const size_t slice_bytes = static_cast<size_t>(pos) * kv_dim * sizeof(float);
            const size_t per_slice_size = sizeof(int32_t) + slice_bytes * 2;
            const size_t packet_size = dist_config.transport->get_packet_size();

            // Phase 2: Receive payload (if any slices)
            struct KvSlice {
                int32_t layer_id;
                const char *key_ptr;
                const char *val_ptr;
            };
            std::vector<KvSlice> incoming_slices;
            std::vector<char> recv_payload;

            if (incoming_hdr.num_slices > 0) {
                const size_t payload_size = incoming_hdr.num_slices * per_slice_size;
                recv_payload.resize(payload_size);
                dist_config.transport->recv_prev(recv_payload.data(), payload_size);

                const char *rp = recv_payload.data();
                for (int i = 0; i < incoming_hdr.num_slices; i++) {
                    KvSlice s;
                    std::memcpy(&s.layer_id, rp, sizeof(s.layer_id));
                    rp += sizeof(s.layer_id);
                    s.key_ptr = rp;
                    rp += slice_bytes;
                    s.val_ptr = rp;
                    rp += slice_bytes;
                    incoming_slices.push_back(s);
                }
            }

            // Extract layers this worker needs
            std::vector<bool> consumed(incoming_slices.size(), false);
            for (size_t i = 0; i < incoming_slices.size(); i++) {
                auto &s = incoming_slices[i];
                if (s.layer_id >= new_start && s.layer_id < new_end) {
                    float *kd = key_cache + static_cast<size_t>(s.layer_id) * seq_len * kv_dim;
                    float *vd = value_cache + static_cast<size_t>(s.layer_id) * seq_len * kv_dim;
                    std::memcpy(kd, s.key_ptr, slice_bytes);
                    std::memcpy(vd, s.val_ptr, slice_bytes);
                    consumed[i] = true;
                }
            }

            // Count forward slices
            int forward_count = 0;
            for (size_t i = 0; i < consumed.size(); i++) {
                if (!consumed[i]) forward_count++;
            }
            for (int l = old_start; l < old_end; l++) {
                if (l < new_start || l >= new_end) forward_count++;
            }

            // Phase 1: Send forward header (padded to packet_size)
            KvTransferHeader fwd_hdr{};
            fwd_hdr.magic = KV_TRANSFER_MAGIC;
            fwd_hdr.num_slices = forward_count;
            fwd_hdr.pos = pos;
            fwd_hdr.kv_dim = kv_dim;

            std::vector<char> fwd_hdr_buf(packet_size, 0);
            std::memcpy(fwd_hdr_buf.data(), &fwd_hdr, sizeof(fwd_hdr));
            dist_config.transport->send_next(fwd_hdr_buf.data(), packet_size);

            // Phase 2: Build and send forward payload
            if (forward_count > 0) {
                const size_t fwd_payload_size = forward_count * per_slice_size;
                std::vector<char> fwd_payload(fwd_payload_size);
                char *wp = fwd_payload.data();

                // Unconsumed incoming slices
                for (size_t i = 0; i < incoming_slices.size(); i++) {
                    if (!consumed[i]) {
                        auto &s = incoming_slices[i];
                        std::memcpy(wp, &s.layer_id, sizeof(s.layer_id));
                        wp += sizeof(s.layer_id);
                        std::memcpy(wp, s.key_ptr, slice_bytes);
                        wp += slice_bytes;
                        std::memcpy(wp, s.val_ptr, slice_bytes);
                        wp += slice_bytes;
                    }
                }

                // Our departing layers
                for (int l = old_start; l < old_end; l++) {
                    if (l < new_start || l >= new_end) {
                        int32_t layer_id = l;
                        std::memcpy(wp, &layer_id, sizeof(layer_id));
                        wp += sizeof(layer_id);
                        float *kp = key_cache + static_cast<size_t>(l) * seq_len * kv_dim;
                        std::memcpy(wp, kp, slice_bytes);
                        wp += slice_bytes;
                        float *vp = value_cache + static_cast<size_t>(l) * seq_len * kv_dim;
                        std::memcpy(wp, vp, slice_bytes);
                        wp += slice_bytes;
                    }
                }

                dist_config.transport->send_next(fwd_payload.data(), fwd_payload_size);
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