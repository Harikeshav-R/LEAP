#ifndef LEAP_KERNEL_TRANSPORT_H
#define LEAP_KERNEL_TRANSPORT_H

#include "Transport.h"
#include <string>

namespace Inference {
    class KernelTransport : public Transport {
    public:
        KernelTransport(std::string dest_ip);
        ~KernelTransport() override;

        void initialize() override;
        void send(const void *data, size_t size) override;
        void recv(void *data, size_t size) override;

    private:
        std::string dest_ip;
        int fd = -1;
        void *mmap_ptr = nullptr;
    };
}

#endif //LEAP_KERNEL_TRANSPORT_H
