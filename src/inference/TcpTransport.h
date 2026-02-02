#ifndef LEAP_TCP_TRANSPORT_H
#define LEAP_TCP_TRANSPORT_H

#include "Transport.h"
#include <string>

namespace Inference {
    class TcpTransport : public Transport {
    public:
        // is_server = true -> Worker (listens)
        // is_server = false -> Master (connects)
        // If next_ip is provided, enables Ring Mode (Listen on port, Connect to next_ip:next_port)
        TcpTransport(std::string ip, int port, bool is_server, std::string next_ip = "", int next_port = 0);

        ~TcpTransport() override;

        void initialize() override;

        void send(const void *data, size_t size) override;

        void recv(void *data, size_t size) override;

    private:
        std::string ip;
        int port;
        bool is_server;
        std::string next_ip;
        int next_port;

        int sockfd = -1;   // Input Listener (if server/ring) or Connection (if client legacy)
        int clientfd = -1; // Accepted Input Connection (if server/ring)
        int connfd = -1;   // Output Connection (if ring)

        void setup_socket_low_latency(int fd);
    };
} // namespace Inference

#endif //LEAP_TCP_TRANSPORT_H