#ifndef LEAP_UDP_TRANSPORT_H
#define LEAP_UDP_TRANSPORT_H

#include "Transport.h"
#include <string>
#include <vector>
#include <netinet/in.h>

namespace Inference {
    class UdpTransport : public Transport {
    public:
        UdpTransport(std::string ip, int port, bool is_server, std::string next_ip = "", int next_port = 0);

        ~UdpTransport() override;

        void initialize() override;

        void send(const void *data, size_t size) override;
        void recv(void *data, size_t size) override;

        void send_next(const void *data, size_t size) override;
        void recv_next(void *data, size_t size) override;
        void send_prev(const void *data, size_t size) override;
        void recv_prev(void *data, size_t size) override;

    private:
        std::string ip;
        int port;
        bool is_server;
        std::string next_ip;
        int next_port;

        int sockfd = -1;
        sockaddr_in prev_addr{};
        sockaddr_in next_addr{};
        bool prev_addr_set = false;
        
        // Sequence ID strategy:
        // Initialized to 0x8000 (32768) to provide a high-entropy start point that minimizes
        // immediate collisions with other nodes (like KernelTransport) that may start at 0.
        // In distributed pipelines with mixed transports, this offset helps the receiving
        // kernel module distinguish new transactions. The ID wraps naturally as uint16_t.
        uint16_t seq_id = 0x8000; 

        // Buffer for reassembly
        std::vector<uint8_t> reassembly_buffer;
    };
}

#endif //LEAP_UDP_TRANSPORT_H