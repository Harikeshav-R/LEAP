#ifndef LEAP_TRANSPORT_H
#define LEAP_TRANSPORT_H

#include <cstddef>

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
    };
} // namespace Inference

#endif //LEAP_TRANSPORT_H