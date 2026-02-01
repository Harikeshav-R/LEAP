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
    KernelTransport::KernelTransport(std::string dest_ip, int port)
        : dest_ip(std::move(dest_ip)), port(port) {
    }

    KernelTransport::~KernelTransport() {
        if (mmap_ptr && mmap_ptr != MAP_FAILED)
            munmap(mmap_ptr, LEAP_BUFFER_SIZE * 2); // Unmap total size
        if (fd != -1) close(fd);
    }

    void KernelTransport::initialize() {
        fd = open("/dev/leap_tensor", O_RDWR);
        if (fd < 0) throw std::runtime_error("Failed to open /dev/leap_tensor. Is the module loaded?");

        // Zero-Copy Map - Map BOTH buffers (RX + TX) = 16MB
        mmap_ptr = mmap(NULL, LEAP_BUFFER_SIZE * 2, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
        if (mmap_ptr == MAP_FAILED) {
            throw std::runtime_error(std::string("Failed to mmap kernel buffer: ") + std::strerror(errno));
        }

        // Set Listening Port
        unsigned short port_short = static_cast<unsigned short>(port);
        if (ioctl(fd, LEAP_IOCTL_SET_PORT, &port_short) < 0) {
            std::cerr << "Warning: Failed to set listening port in kernel" << std::endl;
        }

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
        if (size > LEAP_BUFFER_SIZE) {
            throw std::runtime_error("Send size too large for LEAP buffer (Max: " + std::to_string(LEAP_BUFFER_SIZE) + ")");
        }

        // Zero-Copy Optimization with Double Buffering:
        // Copy to TX buffer partition (Upper 8MB)
        // Offset = LEAP_BUFFER_SIZE
        uint8_t* tx_buffer = static_cast<uint8_t*>(mmap_ptr) + LEAP_BUFFER_SIZE;
        std::memcpy(tx_buffer, data, size);

        // Trigger Send via IOCTL (Kernel reads from Upper 8MB)
        unsigned int len = static_cast<unsigned int>(size);
        if (ioctl(fd, LEAP_IOCTL_SEND, &len) < 0) {
            throw std::runtime_error("Kernel send IOCTL failed");
        }
    }

    void KernelTransport::recv(void *data, const size_t size) {
        // Wait for data to be ready (Kernel writes to Lower 8MB)
        if (ioctl(fd, LEAP_IOCTL_WAIT_DATA, 0) < 0) {
            throw std::runtime_error("IOCTL wait failed");
        }

        if (size > LEAP_BUFFER_SIZE) throw std::runtime_error("Recv size too large for kernel buffer");

        // Read from RX buffer partition (Lower 8MB)
        // Offset = 0
        std::memcpy(data, mmap_ptr, size);
    }
}