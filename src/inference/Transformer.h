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
        Transport *transport = nullptr;
    };

    class Transformer {
    public:
        Config config{};
        DistributedConfig dist_config{};

        virtual ~Transformer() = default;

        // The core function: forward pass
        virtual float *forward(int token, int pos) = 0;

        // Worker loop: receive tensor, process layers, send back
        virtual void worker_loop() = 0;

        void set_distributed_config(DistributedMode mode, int split_layer, Transport *transport) {
            dist_config.mode = mode;
            dist_config.split_layer = split_layer;
            dist_config.transport = transport;
        }

        // Factory method to create the appropriate Transformer (Float or Quantized) based on file
        static std::unique_ptr<Transformer> create(const std::string &checkpoint_path);
    };
} // namespace Inference


#endif // LEAP_TRANSFORMER_H