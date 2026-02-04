#ifndef LEAP_TRANSFORMER_H
#define LEAP_TRANSFORMER_H

#include "Config.h"
#include "Transport.h"
#include "SystemMonitor.h"
#include <string>
#include <memory>
#include <vector>
#include <iostream>

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

    enum class PacketType : uint32_t {
        Inference = 0,
        StatsUpdate = 1,
        ConfigUpdate = 2
    };

    struct PacketHeader {
        uint32_t magic;      // 0x4C454150
        PacketType type;     // Message Type
        uint32_t pos;        // Token Position
        uint32_t flags;      // Flags
        uint32_t payload_size;
        uint32_t reserved[3]; // Padding to 32 bytes
    };

    constexpr int FLAG_NO_REPLY = 0;
    constexpr int FLAG_NEED_REPLY = 1;

    // Config Packet Structure (Payload for ConfigUpdate)
    struct LayerConfig {
        int split_layer;
        int end_layer;
    };

    class Transformer {
    public:
        Config config{};
        DistributedConfig dist_config{};
        SystemMonitor monitor{};

        virtual ~Transformer() = default;

        // The core function: forward pass
        virtual float *forward(int token, int pos, int flags = FLAG_NEED_REPLY) = 0;

        // Worker loop: receive tensor, process layers, send back
        virtual void worker_loop() = 0;

        void set_distributed_config(const DistributedConfig &config) {
            dist_config = config;
        }

        virtual void update_config(int split, int end) {
            dist_config.split_layer = split;
            dist_config.end_layer = end;
            dist_config.is_tail = (end == config.n_layers);
            std::cout << "[Config] Updated: Layers " << split << " -> " << end << std::endl;
        }

        virtual void distribute_config(const std::vector<LayerConfig> &configs) {
            (void)configs; 
        }

        virtual std::vector<NodeStats> collect_stats() {
            return {};
        }

        // Factory method to create the appropriate Transformer (Float or Quantized) based on file
        static std::unique_ptr<Transformer> create(const std::string &checkpoint_path);

    protected:
        // Optimization: Reusable buffer for network transfers to avoid repeated allocations
        std::vector<char> transfer_buffer;
    };
} // namespace Inference


#endif // LEAP_TRANSFORMER_H