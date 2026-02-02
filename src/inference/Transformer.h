#ifndef LEAP_TRANSFORMER_H
#define LEAP_TRANSFORMER_H

#include "Config.h"
#include "Transport.h"
#include <string>
#include <memory>

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
        std::string next_ip;
        int next_port = 0;
        bool is_tail = false;
    };

    struct PacketHeader {
        int pos;
        int flags; // 0: No Reply (Fire-and-Forget), 1: Need Reply
    };
    const int FLAG_NO_REPLY = 0;
    const int FLAG_NEED_REPLY = 1;

    class Transformer {
    public:
        Config config{};
        DistributedConfig dist_config{};

        virtual ~Transformer() = default;

        // The core function: forward pass
        virtual float *forward(int token, int pos, int flags = FLAG_NEED_REPLY) = 0;

        // Worker loop: receive tensor, process layers, send back
        virtual void worker_loop() = 0;

        void set_distributed_config(DistributedConfig config) {
            dist_config = config;
        }

        // Factory method to create the appropriate Transformer (Float or Quantized) based on file
        static std::unique_ptr<Transformer> create(const std::string &checkpoint_path);
    };
} // namespace Inference


#endif // LEAP_TRANSFORMER_H