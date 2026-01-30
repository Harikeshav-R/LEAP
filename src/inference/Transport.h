#ifndef LEAP_TRANSPORT_H
#define LEAP_TRANSPORT_H

#include <cstddef>

namespace Inference {
    class Transport {
    public:
        virtual ~Transport() = default;

        // Initialize the connection (Client connects, Server binds/listens)
        virtual void initialize() = 0;

        // Send buffer
        virtual void send(const void *data, size_t size) = 0;

        // Receive into buffer
        virtual void recv(void *data, size_t size) = 0;
    };
} // namespace Inference

#endif //LEAP_TRANSPORT_H