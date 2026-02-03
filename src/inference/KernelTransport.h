#ifndef LEAP_KERNEL_TRANSPORT_H
#define LEAP_KERNEL_TRANSPORT_H

#include "Transport.h"
#include <string>

namespace Inference {
    class KernelTransport : public Transport {
    public:
        // Ring Topology:
        // - Binds to `port` (Ingress)
        // - Sends to `next_ip:next_port` (Egress)
        KernelTransport(int port, std::string next_ip, int next_port);

        ~KernelTransport() override;

        void initialize() override;

        void send(const void *data, size_t size) override;

        void recv(void *data, size_t size) override;

        void send_next(const void *data, size_t size) override;

        void recv_next(void *data, size_t size) override; // Not used
        void send_prev(const void *data, size_t size) override; // Not used
        void recv_prev(void *data, size_t size) override;

        void send_multipart_next(const void *header, size_t header_size, const void *data, size_t data_size) override;

    private:
        void set_destination(const std::string &ip, int target_port) const;

        void recv_internal(void *data, size_t size, bool update_prev);

        int port;
        int fd = -1;
        void *mmap_ptr = nullptr;

        std::string prev_ip;
        int prev_port;

        std::string next_ip;
        int next_port;
    };
}

#endif //LEAP_KERNEL_TRANSPORT_H