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
    KernelTransport::KernelTransport(std::string dest_ip, int port, std::string next_ip, int next_port)
        : dest_ip(std::move(dest_ip)), port(port), prev_ip(this->dest_ip), prev_port(port), 
          next_ip(std::move(next_ip)), next_port(next_port) {
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

        std::cout << "KernelTransport: Initialized. Banks=" << LEAP_RX_BANKS 
                  << " BankSize=" << LEAP_RX_BANK_SIZE << " bytes." << std::endl;

        // Set Listening Port
        unsigned short port_short = static_cast<unsigned short>(port);
        if (ioctl(fd, LEAP_IOCTL_SET_PORT, &port_short) < 0) {
            std::cerr << "Warning: Failed to set listening port in kernel" << std::endl;
        }

        // Set Default Destination (Prev)
        set_destination(prev_ip, prev_port);
    }

    void KernelTransport::set_destination(const std::string& ip, int target_port) {
        unsigned int ip_int;
        if (inet_pton(AF_INET, ip.c_str(), &ip_int) <= 0) {
            throw std::runtime_error("Invalid IP for KernelTransport: " + ip);
        }

        if (ioctl(fd, LEAP_IOCTL_SET_DEST, &ip_int) < 0) {
            std::cerr << "Warning: Failed to set destination IP in kernel" << std::endl;
        }
        
        if (target_port > 0) {
            unsigned short p = static_cast<unsigned short>(target_port);
            if (ioctl(fd, LEAP_IOCTL_SET_TX_PORT, &p) < 0) {
                 std::cerr << "Warning: Failed to set TX port in kernel" << std::endl;
            }
        }
    }

    void KernelTransport::send(const void *data, const size_t size) {
        if (size > LEAP_BUFFER_SIZE) {
            throw std::runtime_error("Send size too large for LEAP buffer (Max: " + std::to_string(LEAP_BUFFER_SIZE) + ")");
        }

        // Zero-Copy Optimization with Double Buffering:
        // Copy to TX buffer partition (Upper 8MB)
        uint8_t* tx_buffer = static_cast<uint8_t*>(mmap_ptr) + LEAP_BUFFER_SIZE;
        std::memcpy(tx_buffer, data, size);

        // Trigger Send via IOCTL (Kernel reads from Upper 8MB)
        unsigned int len = static_cast<unsigned int>(size);
        if (ioctl(fd, LEAP_IOCTL_SEND, &len) < 0) {
            throw std::runtime_error("Kernel send IOCTL failed");
        }
    }

    void KernelTransport::recv_internal(void *data, const size_t size, bool update_prev) {
        // Wait for data to be ready (Kernel writes to Lower 8MB, split into 64 banks)
        // IOCTL returns the bank index (0 to 63)
        int bank = ioctl(fd, LEAP_IOCTL_WAIT_DATA, 0);
        if (bank < 0) {
            throw std::runtime_error("IOCTL wait failed");
        }

        if (update_prev) {
            // Learn Source IP/Port for this bank
            leap_bank_metadata meta;
            meta.bank_idx = bank;
            if (ioctl(fd, LEAP_IOCTL_GET_BANK_SRC, &meta) >= 0) {
                char ip_str[INET_ADDRSTRLEN];
                if (inet_ntop(AF_INET, &meta.saddr, ip_str, INET_ADDRSTRLEN)) {
                    if (this->prev_ip != ip_str) {
                        std::cout << "KernelTransport: Learned Prev IP: " << ip_str << " (Bank " << bank << ")" << std::endl;
                        this->prev_ip = ip_str;
                    }
                    this->prev_port = ntohs(meta.sport);
                }
            }
        }

        if (size > LEAP_RX_BANK_SIZE) throw std::runtime_error("Recv size too large for kernel bank");

        // Read from RX buffer partition (Bank 0 to 63)
        // Bank Size = 128KB (LEAP_BUFFER_SIZE / 64)
        size_t offset = bank * LEAP_RX_BANK_SIZE;
        std::memcpy(data, static_cast<uint8_t*>(mmap_ptr) + offset, size);
    }

    void KernelTransport::recv(void *data, const size_t size) {
        recv_internal(data, size, false); // Default recv doesn't update prev by default safely
    }

    void KernelTransport::send_next(const void *data, size_t size) {
        if (next_ip.empty()) throw std::runtime_error("Next IP not configured for KernelTransport");
        set_destination(next_ip, next_port);
        send(data, size);
    }

    void KernelTransport::recv_next(void *data, size_t size) {
        // Recv from next node - DO NOT update prev_ip
        recv_internal(data, size, false);
    }

    void KernelTransport::send_prev(const void *data, size_t size) {
        // Explicitly set port to prev_port. Relying on kernel's learned port is unsafe
        // because send_next() overwrites the global dest_port in the kernel module.
        // This assumes symmetric port configuration (Prev listens on prev_port).
        set_destination(prev_ip, prev_port); 
        send(data, size);
    }

    void KernelTransport::recv_prev(void *data, size_t size) {
        // Recv from prev node - Update prev_ip
        recv_internal(data, size, true);
    }
}