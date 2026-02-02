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
        std::string dest_ip; // Current Dest IP
        int port;            // Listening Port

        std::string prev_ip; // Usually same as initial dest_ip
        int prev_port;       // Usually same as initial port (symmetric) or needs explicit config? 
                             // Wait, port in constructor is LISTEN port. 
                             // We need to know TARGET port for Prev.
                             // Let's assume symmetric ports for now or use the 'port' arg as target too?
                             // Standard UdpTransport uses 'port' for both if not specified.
                             // Let's stick to the constructor signature I updated.

        std::string next_ip;
        int next_port;

        int fd = -1;
        void *mmap_ptr = nullptr;

        void set_destination(const std::string& ip, int port);
    };
}

#endif //LEAP_KERNEL_TRANSPORT_H