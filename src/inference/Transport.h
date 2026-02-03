#ifndef LEAP_TRANSPORT_H
#define LEAP_TRANSPORT_H

#include <cstddef>
#include <vector>
#include <cstring>

namespace Inference {
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
        virtual void send_next(const void *data, size_t size) { send(data, size); }
        virtual void recv_next(void *data, size_t size) { recv(data, size); }
        virtual void send_prev(const void *data, size_t size) { send(data, size); }
        virtual void recv_prev(void *data, size_t size) { recv(data, size); }

        // Optimization: Zero-Copy Send (Header + Payload)
        virtual void send_multipart_next(const void* header, size_t header_size, const void* data, size_t data_size) {
            // Fallback: This effectively mimics the old behavior if not overridden
            // But allows optimized transports to avoid the merge copy.
            std::vector<char> buffer(header_size + data_size);
            std::memcpy(buffer.data(), header, header_size);
            std::memcpy(buffer.data() + header_size, data, data_size);
            send_next(buffer.data(), buffer.size());
        }
    };
} // namespace Inference

#endif //LEAP_TRANSPORT_H