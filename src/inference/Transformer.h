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

        // Clear KV cache (needed after layer redistribution to avoid stale state)
        virtual void clear_kv_cache() = 0;

        // Factory method to create the appropriate Transformer (Float or Quantized) based on file
        static std::unique_ptr<Transformer> create(const std::string &checkpoint_path);

    protected:
        // Optimization: Reusable buffer for network transfers to avoid repeated allocations
        std::vector<char> transfer_buffer;
    };
} // namespace Inference


#endif // LEAP_TRANSFORMER_H