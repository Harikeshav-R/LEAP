#ifndef LEAP_UDP_TRANSPORT_H
#define LEAP_UDP_TRANSPORT_H

#include "Transport.h"
#include <string>
#include <vector>
#include <netinet/in.h>

namespace Inference {
    class UdpTransport : public Transport {
    public:
        // If next_ip is provided, enables Ring Mode (Send to next_ip:next_port)
        UdpTransport(std::string ip, int port, bool is_server, std::string next_ip = "", int next_port = 0);

        ~UdpTransport() override;

        void initialize() override;

        void send(const void *data, size_t size) override;

        void recv(void *data, size_t size) override;

    private:
        std::string ip;
        int port;
        bool is_server;
        std::string next_ip;
        int next_port;

        int sockfd = -1;
        sockaddr_in dest_addr{};
        uint16_t seq_id = 0;

        // Buffer for reassembly
        std::vector<uint8_t> reassembly_buffer;
    };
}

#endif //LEAP_UDP_TRANSPORT_H