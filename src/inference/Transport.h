#ifndef LEAP_TRANSPORT_H
#define LEAP_TRANSPORT_H

#include <cstddef>
#include <vector>
#include <cstring>
#include <cstdint>

namespace Inference {
    // Control message types for dynamic layer resizing
    enum class ControlMessageType : uint8_t {
        RESIZE_LAYERS = 1,  // Single worker resize (legacy)
        RESIZE_CHAIN = 2,   // Multi-worker chain resize (new)
        ACK = 3
    };

    // Maximum workers supported in a chain
    constexpr int MAX_WORKERS = 16;

    // Layer range for a single worker
    struct LayerRange {
        int start_layer;
        int end_layer;
        bool is_tail;
    };

    struct ControlMessage {
        ControlMessageType type;
        int split_layer;      // For single-worker mode: worker start layer
        int end_layer;        // For single-worker mode: worker end layer
        bool is_tail;         // For single-worker mode
        
        // Multi-worker chain fields (used with RESIZE_CHAIN)
        int worker_index;     // Current worker's index (0-based)
        int total_workers;    // Total number of workers
        LayerRange ranges[MAX_WORKERS];  // Layer ranges for all workers
    };

    // Magic marker to distinguish control packets from data packets
    constexpr uint16_t CONTROL_MAGIC = 0xC0DE;

    struct ControlPacketHeader {
        uint16_t magic;  // CONTROL_MAGIC
        ControlMessage msg;
    } __attribute__((packed));

    class Transport {
    public:
        virtual ~Transport() = default;

        // Initialize the connection (Client connects, Server binds/listens)
        virtual void initialize() = 0;

        // Send buffer (Legacy/Simple)
        virtual void send(const void *data, size_t size) = 0;

        // Receive into buffer (Legacy/Simple)
        virtual void recv(void *data, size_t size) = 0;

        // Chain Methods
        virtual void send_next(const void *data, const size_t size) { send(data, size); }
        virtual void recv_next(void *data, const size_t size) { recv(data, size); }
        virtual void send_prev(const void *data, const size_t size) { send(data, size); }
        virtual void recv_prev(void *data, const size_t size) { recv(data, size); }

        // Optimization: Zero-Copy Send (Header + Payload)
        virtual void send_multipart_next(const void *header, const size_t header_size, const void *data,
                                         const size_t data_size) {
            // Fallback: This effectively mimics the old behavior if not overridden
            // But allows optimized transports to avoid the merge copy.
            std::vector<char> buffer(header_size + data_size);
            std::memcpy(buffer.data(), header, header_size);
            std::memcpy(buffer.data() + header_size, data, data_size);
            send_next(buffer.data(), buffer.size());
        }

        // Control channel for layer resizing commands
        virtual void send_control(const ControlMessage &msg) {
            ControlPacketHeader pkt{CONTROL_MAGIC, msg};
            send_next(&pkt, sizeof(pkt));
        }

        // Non-blocking check for control messages. Returns true if a message was received.
        virtual bool recv_control_nonblocking(ControlMessage &msg) {
            (void)msg;
            return false;  // Default: no control channel support
        }
    };
} // namespace Inference

#endif //LEAP_TRANSPORT_H