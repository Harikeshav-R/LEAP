#include "KernelTransport.h"
#include "../kernel/leap_protocol.h"
#include <fcntl.h>
#include <unistd.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <arpa/inet.h>
#include <stdexcept>
#include <iostream>
#include <cstring>

namespace Inference {
    KernelTransport::KernelTransport(std::string dest_ip) : dest_ip(std::move(dest_ip)) {}

    KernelTransport::~KernelTransport() {
        if (mmap_ptr && mmap_ptr != MAP_FAILED) munmap(mmap_ptr, LEAP_BUFFER_SIZE);
        if (fd != -1) close(fd);
    }

    void KernelTransport::initialize() {
        fd = open("/dev/leap_tensor", O_RDWR);
        if (fd < 0) throw std::runtime_error("Failed to open /dev/leap_tensor. Is the module loaded?");

        // Zero-Copy Map
        mmap_ptr = mmap(NULL, LEAP_BUFFER_SIZE, PROT_READ, MAP_SHARED, fd, 0);
        if (mmap_ptr == MAP_FAILED) throw std::runtime_error("Failed to mmap kernel buffer");

        // Configure Destination IP for Sending
        unsigned int ip_int;
        if (inet_pton(AF_INET, dest_ip.c_str(), &ip_int) <= 0) {
             throw std::runtime_error("Invalid IP for KernelTransport");
        }
        
        if (ioctl(fd, LEAP_IOCTL_SET_DEST, &ip_int) < 0) {
            std::cerr << "Warning: Failed to set destination IP in kernel" << std::endl;
        }
    }

    void KernelTransport::send(const void *data, const size_t size) {
        // Just write to the device node. The kernel handles fragmentation and UDP sending.
        ssize_t ret = ::write(fd, data, size);
        if (ret < 0) throw std::runtime_error("Kernel send failed");
    }

    void KernelTransport::recv(void *data, const size_t size) {
        // Wait for data to be ready in the kernel buffer
        if (ioctl(fd, LEAP_IOCTL_WAIT_DATA, 0) < 0) {
            throw std::runtime_error("IOCTL wait failed");
        }

        // Copy from mmap buffer (Zero-Copy read from network standpoint, technically 1 copy here to dest)
        // Optimization: In a real system, we would parse the tensor directly from mmap_ptr without copying.
        if (size > LEAP_BUFFER_SIZE) throw std::runtime_error("Recv size too large for kernel buffer");
        std::memcpy(data, mmap_ptr, size);
    }
}
