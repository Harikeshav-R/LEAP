#ifndef LEAP_TRANSPORT_H
#define LEAP_TRANSPORT_H

#include <cstddef>
#include <vector>
#include <cstring>
#include <cstdint>

namespace Inference {
    // Control message types for dynamic layer resizing
    enum class ControlMessageType : uint8_t {
        RESIZE_LAYERS = 1,  // Single worker resize
        RESIZE_CHAIN = 2,   // Multi-worker chain resize
        ACK = 3
    };

    // Maximum workers supported in a chain
    constexpr int MAX_WORKERS = 16;

    // Layer range for a single worker (packed for network transmission)
    struct LayerRange {
        int32_t start_layer;
        int32_t end_layer;
        uint8_t is_tail;
    } __attribute__((packed));

    // Control message for layer resizing (packed for consistent sizing)
    struct ControlMessage {
        ControlMessageType type;
        int32_t split_layer;      // For single-worker mode: worker start layer
        int32_t end_layer;        // For single-worker mode: worker end layer
        uint8_t is_tail;          // For single-worker mode
        
        // Multi-worker chain fields (used with RESIZE_CHAIN)
        int32_t worker_index;     // Current worker's index (0-based)
        int32_t total_workers;    // Total number of workers
        LayerRange ranges[MAX_WORKERS];  // Layer ranges for all workers
    } __attribute__((packed));

    // Magic marker to distinguish control packets from data packets
    constexpr uint16_t CONTROL_MAGIC = 0xC0DE;

    // Control packet header (packed for network transmission)
    struct ControlPacketHeader {
        uint16_t magic;  // CONTROL_MAGIC
        ControlMessage msg;
    } __attribute__((packed));

    class Transport {
    protected:
        size_t packet_size_ = 0;  // Data packet size (header + dim * sizeof(float))
        
    public:
        virtual ~Transport() = default;

        // Set the expected packet size for control message padding
        void set_packet_size(size_t size) { packet_size_ = size; }
        size_t get_packet_size() const { return packet_size_; }

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
        // Pads message to packet_size_ so it flows through recv_prev correctly
        virtual void send_control(const ControlMessage &msg) {
            ControlPacketHeader pkt{CONTROL_MAGIC, msg};
            
            if (packet_size_ > 0 && packet_size_ >= sizeof(pkt)) {
                // Pad to full packet size so recv_prev receives it correctly
                std::vector<char> buffer(packet_size_, 0);
                std::memcpy(buffer.data(), &pkt, sizeof(pkt));
                send_next(buffer.data(), buffer.size());
            } else {
                // Fallback: send raw (may not work with blocking recv)
                send_next(&pkt, sizeof(pkt));
            }
        }

        // Non-blocking check for control messages. Returns true if a message was received.
        virtual bool recv_control_nonblocking(ControlMessage &msg) {
            (void)msg;
            return false;  // Default: no control channel support
        }
    };
} // namespace Inference

#endif //LEAP_TRANSPORT_H