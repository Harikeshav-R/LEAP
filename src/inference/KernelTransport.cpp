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
    KernelTransport::KernelTransport(const int port, std::string next_ip, const int next_port)
        : port(port), prev_port(0),
          next_ip(std::move(next_ip)), next_port(next_port) {
    }

    KernelTransport::~KernelTransport() {
        if (mmap_ptr &&mmap_ptr 
        !=
        MAP_FAILED
        )
        munmap(mmap_ptr, LEAP_BUFFER_SIZE * 2); // Unmap total size
        if (fd != -1) close(fd);
    }

    void KernelTransport::initialize() {
        fd = open("/dev/leap_tensor", O_RDWR);
        if (fd < 0) throw std::runtime_error("Failed to open /dev/leap_tensor. Is the module loaded?");

        // Zero-Copy Map - Map BOTH buffers (RX + TX) = 16MB
        mmap_ptr = mmap(nullptr, LEAP_BUFFER_SIZE * 2, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
        if (mmap_ptr == MAP_FAILED) {
            throw std::runtime_error(std::string("Failed to mmap kernel buffer: ") + std::strerror(errno));
        }

        std::cout << "KernelTransport: Initialized. Banks=" << LEAP_RX_BANKS
                << " BankSize=" << LEAP_RX_BANK_SIZE << " bytes." << std::endl;

        // Set Listening Port
        auto port_short = static_cast<unsigned short>(port);
        if (ioctl(fd, LEAP_IOCTL_SET_PORT, &port_short) < 0) {
            std::cerr << "Warning: Failed to set listening port in kernel" << std::endl;
        }

        // Only set destination if prev_ip is known (e.g. hardcoded or learned)
        if (!prev_ip.empty()) {
            set_destination(prev_ip, prev_port);
        }
    }

    void KernelTransport::set_destination(const std::string &ip, const int target_port) const {
        unsigned int ip_int;
        if (inet_pton(AF_INET, ip.c_str(), &ip_int) <= 0) {
            throw std::runtime_error("Invalid IP for KernelTransport: " + ip);
        }

        if (ioctl(fd, LEAP_IOCTL_SET_DEST, &ip_int) < 0) {
            std::cerr << "Warning: Failed to set destination IP in kernel" << std::endl;
        }

        if (target_port > 0) {
            auto p = static_cast<unsigned short>(target_port);
            if (ioctl(fd, LEAP_IOCTL_SET_TX_PORT, &p) < 0) {
                std::cerr << "Warning: Failed to set TX port in kernel" << std::endl;
            }
        }
    }

    void KernelTransport::send(const void *data, const size_t size) {
        // Default send -> send_next
        send_next(data, size);
    }

    void KernelTransport::recv_internal(void *data, const size_t size, const bool update_prev) {
        // Wait for data to be ready (Kernel writes to Lower 8MB, split into 64 banks)
        // IOCTL returns the bank index (0 to 63)
        const int bank = ioctl(fd, LEAP_IOCTL_WAIT_DATA, 0);
        if (bank < 0) {
            throw std::runtime_error("IOCTL wait failed");
        }

        if (update_prev) {
            // Learn Source IP/Port for this bank
            leap_bank_metadata meta{};
            meta.bank_idx = bank;
            if (ioctl(fd, LEAP_IOCTL_GET_BANK_SRC, &meta) >= 0) {
                char ip_str[INET_ADDRSTRLEN];
                if (inet_ntop(AF_INET, &meta.saddr, ip_str, INET_ADDRSTRLEN)) {
                    if (this->prev_ip != ip_str) {
                        std::cout << "KernelTransport: Learned Prev IP: " << ip_str << " (Bank " << bank << ")" <<
                                std::endl;
                        this->prev_ip = ip_str;
                    }
                    this->prev_port = ntohs(meta.sport);
                }
            }
        }

        if (size > LEAP_RX_BANK_SIZE) throw std::runtime_error("Recv size too large for kernel bank");

        // Read from RX buffer partition (Bank 0 to 63)
        // Bank Size = 128KB (LEAP_BUFFER_SIZE / 64)
        const size_t offset = bank * LEAP_RX_BANK_SIZE;
        std::memcpy(data, static_cast<uint8_t *>(mmap_ptr) + offset, size);
    }

    void KernelTransport::recv(void *data, const size_t size) {
        recv_prev(data, size);
    }

    void KernelTransport::send_next(const void *data, const size_t size) {
        if (next_ip.empty()) throw std::runtime_error("Next IP not configured for KernelTransport");
        if (size > LEAP_BUFFER_SIZE) {
            throw std::runtime_error("Send size too large for LEAP buffer");
        }
        set_destination(next_ip, next_port);

        // Zero-Copy Optimization with Double Buffering:
        // Copy to TX buffer partition (Upper 8MB)
        uint8_t *tx_buffer = static_cast<uint8_t *>(mmap_ptr) + LEAP_BUFFER_SIZE;
        std::memcpy(tx_buffer, data, size);

        // Trigger Send via IOCTL (Kernel reads from Upper 8MB)
        auto len = static_cast<unsigned int>(size);
        if (ioctl(fd, LEAP_IOCTL_SEND, &len) < 0) {
            throw std::runtime_error("Kernel send IOCTL failed");
        }
    }

    void KernelTransport::send_multipart_next(const void *header, const size_t header_size, const void *data,
                                              const size_t data_size) {
        if (next_ip.empty()) throw std::runtime_error("Next IP not configured for KernelTransport");

        const size_t total_size = header_size + data_size;
        if (total_size > LEAP_BUFFER_SIZE) {
            throw std::runtime_error("Send size too large for LEAP buffer");
        }
        set_destination(next_ip, next_port);

        const auto tx_buffer = static_cast<uint8_t *>(mmap_ptr) + LEAP_BUFFER_SIZE;

        // Scatter/Gather Copy into contiguous kernel buffer
        std::memcpy(tx_buffer, header, header_size);
        std::memcpy(tx_buffer + header_size, data, data_size);

        auto len = static_cast<unsigned int>(total_size);
        if (ioctl(fd, LEAP_IOCTL_SEND, &len) < 0) {
            throw std::runtime_error("Kernel send IOCTL failed");
        }
    }

    void KernelTransport::recv_next(void *data, size_t size) {
        throw std::runtime_error("recv_next not supported in Ring");
    }

    void KernelTransport::send_prev(const void *data, size_t size) {
        throw std::runtime_error("send_prev not supported in Ring");
    }

    void KernelTransport::recv_prev(void *data, const size_t size) {
        // Recv from prev node - Update prev_ip
        recv_internal(data, size, true);
    }

    void KernelTransport::send_control(const ControlMessage &msg) {
        if (next_ip.empty()) throw std::runtime_error("Next IP not configured for KernelTransport control");

        // For kernel transport, control messages use the same mechanism as regular data
        // but are small enough to fit in a single packet
        ControlPacketHeader pkt{CONTROL_MAGIC, msg};
        send_next(&pkt, sizeof(pkt));
    }

    bool KernelTransport::recv_control_nonblocking(ControlMessage &msg) {
        // Kernel transport uses blocking ioctl for data receive
        // For control messages, we can't easily do non-blocking check without modifying kernel module
        // For now, control messages for kernel transport are checked inline in worker_loop
        // by inspecting the first bytes of received data before processing
        (void)msg;
        return false;  // Non-blocking control not directly supported; handled at higher level
    }
}