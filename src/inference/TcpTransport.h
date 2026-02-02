#ifndef LEAP_TCP_TRANSPORT_H
#define LEAP_TCP_TRANSPORT_H

#include "Transport.h"
#include <string>

namespace Inference {
    class TcpTransport : public Transport {
    public:
        // is_server = true -> Worker (listens on ip:port, optionally connects to next_ip:next_port)
        // is_server = false -> Master (connects to ip:port)
        TcpTransport(std::string ip, int port, bool is_server, std::string next_ip = "", int next_port = 0);

        ~TcpTransport() override;

        void initialize() override;

        void send(const void *data, size_t size) override; // Legacy/Context-aware
        void recv(void *data, size_t size) override;       // Legacy/Context-aware

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

        int sockfd = -1;    // Listening socket (if server) or Egress socket (if master)
        int ingress_fd = -1; // Incoming connection from Previous (if server)
        int egress_fd = -1;  // Outgoing connection to Next (if worker has next, or master)

        void setup_socket_low_latency(int fd);
    };
} // namespace Inference

#endif //LEAP_TCP_TRANSPORT_H