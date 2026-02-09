#ifndef LEAP_TCP_TRANSPORT_H
#define LEAP_TCP_TRANSPORT_H

#include "Transport.h"
#include <string>

namespace Inference {
    class TcpTransport : public Transport {
    public:
        // Symmetric Ring Topology:
        // - Binds to `port` (Ingress from Prev)
        // - Connects to `next_ip:next_port` (Egress to Next)
        TcpTransport(std::string ip, int port, std::string next_ip, int next_port);

        ~TcpTransport() override;

        void initialize() override;

        void send(const void *data, size_t size) override; // Default: send_next
        void recv(void *data, size_t size) override; // Default: recv_prev

        void send_next(const void *data, size_t size) override;

        void recv_next(void *data, size_t size) override;

        void send_prev(const void *data, size_t size) override;

        void recv_prev(void *data, size_t size) override;

        void send_multipart_next(const void *header, size_t header_size, const void *data, size_t data_size) override;

        // Control channel for dynamic layer resizing
        void send_control(const ControlMessage &msg) override;
        bool recv_control_nonblocking(ControlMessage &msg) override;

    private:
        std::string ip;
        int port;
        std::string next_ip;
        int next_port;

        int sockfd = -1; // Listening socket
        int ingress_fd = -1; // Incoming connection from Previous
        int egress_fd = -1; // Outgoing connection to Next

        static void setup_socket_low_latency(int fd);
    };
} // namespace Inference

#endif //LEAP_TCP_TRANSPORT_H