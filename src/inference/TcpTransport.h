#ifndef LEAP_TCP_TRANSPORT_H
#define LEAP_TCP_TRANSPORT_H

#include "Transport.h"
#include <string>

namespace Inference {
    class TcpTransport : public Transport {
    public:
        // is_server = true -> Worker (listens)
        // is_server = false -> Master (connects)
        TcpTransport(std::string ip, int port, bool is_server);

        ~TcpTransport() override;

        void initialize() override;

        void send(const void *data, size_t size) override;

        void recv(void *data, size_t size) override;

    private:
        std::string ip;
        int port;
        bool is_server;
        int sockfd = -1;
        int clientfd = -1; // For server mode, the connected client

        void setup_socket_low_latency(int fd);
    };
} // namespace Inference

#endif //LEAP_TCP_TRANSPORT_H