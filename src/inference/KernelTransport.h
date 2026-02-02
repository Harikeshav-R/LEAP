#ifndef LEAP_KERNEL_TRANSPORT_H
#define LEAP_KERNEL_TRANSPORT_H

#include "Transport.h"
#include <string>

namespace Inference {
    class KernelTransport : public Transport {
    public:
        // dest_ip/port are initial defaults (usually Prev/Master)
        KernelTransport(std::string dest_ip, int port, std::string next_ip = "", int next_port = 0);

        ~KernelTransport() override;

        void initialize() override;

        void send(const void *data, size_t size) override;
        void recv(void *data, size_t size) override;

        void send_next(const void *data, size_t size) override;
        void recv_next(void *data, size_t size) override;
        void send_prev(const void *data, size_t size) override;
        void recv_prev(void *data, size_t size) override;

    private:
        void recv_internal(void *data, size_t size, bool update_prev);

        std::string dest_ip; // Current Dest IP
        int port;            // Listening Port

        std::string prev_ip; // IP of the previous node in the pipeline
        int prev_port;       // Target port for the previous node

        std::string next_ip;
        int next_port;

        int fd = -1;
        void *mmap_ptr = nullptr;

        void set_destination(const std::string& ip, int port);
    };
}

#endif //LEAP_KERNEL_TRANSPORT_H