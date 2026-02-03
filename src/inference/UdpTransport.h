#ifndef LEAP_UDP_TRANSPORT_H
#define LEAP_UDP_TRANSPORT_H

#include "Transport.h"
#include <string>
#include <vector>
#include <netinet/in.h>

namespace Inference {
    class UdpTransport : public Transport {
    public:
        // Ring Topology:
        // - Binds to `port` (Ingress)
        // - Sends to `next_ip:next_port` (Egress)
        UdpTransport(std::string ip, int port, std::string next_ip, int next_port);

        ~UdpTransport() override;

        void initialize() override;

        void send(const void *data, size_t size) override; // Default: send_next
        void recv(void *data, size_t size) override;       // Default: recv_prev

        void send_next(const void *data, size_t size) override;
        void recv_next(void *data, size_t size) override; // Not used
        void send_prev(const void *data, size_t size) override; // Not used
        void recv_prev(void *data, size_t size) override;

    private:
        std::string ip;
        int port;
        std::string next_ip;
        int next_port;

        int sockfd = -1;
        sockaddr_in prev_addr{};
        sockaddr_in next_addr{};
        bool prev_addr_set = false;
        
        uint16_t seq_id = 0x8000; 

        // Buffer for reassembly
        std::vector<uint8_t> reassembly_buffer;
        // Optimization: Reusable packet buffer
        std::vector<uint8_t> packet_buffer;
    };
}

#endif //LEAP_UDP_TRANSPORT_H