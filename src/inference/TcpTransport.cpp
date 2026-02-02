#include "TcpTransport.h"
#include <sys/socket.h>
#include <arpa/inet.h>
#include <unistd.h>
#include <netinet/tcp.h>
#include <stdexcept>
#include <iostream>
#include <cstring>

namespace Inference {
    TcpTransport::TcpTransport(std::string ip, const int port, const bool is_server, std::string next_ip, int next_port)
        : ip(std::move(ip)), port(port), is_server(is_server), next_ip(std::move(next_ip)), next_port(next_port) {
    }

    TcpTransport::~TcpTransport() {
        if (ingress_fd != -1) close(ingress_fd);
        if (egress_fd != -1) close(egress_fd);
        if (sockfd != -1) close(sockfd);
    }

    void TcpTransport::setup_socket_low_latency(const int fd) {
        int flag = 1;
        // Disable Nagle's algorithm for low latency
        if (setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, reinterpret_cast<char *>(&flag), sizeof(int)) < 0) {
            std::cerr << "Warning: Failed to set TCP_NODELAY" << std::endl;
        }
    }

    void TcpTransport::initialize() {
        sockfd = socket(AF_INET, SOCK_STREAM, 0);
        if (sockfd < 0) {
            throw std::runtime_error("Failed to create socket");
        }

        if (is_server) {
            // Server Mode (Worker)
            // 1. Bind and Listen (Ingress)
            sockaddr_in serv_addr{};
            serv_addr.sin_family = AF_INET;
            serv_addr.sin_port = htons(port);
            serv_addr.sin_addr.s_addr = INADDR_ANY; // Bind to all interfaces

            int opt = 1;
            if (setsockopt(sockfd, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt)) < 0) {
                throw std::runtime_error("setsockopt failed");
            }

            if (bind(sockfd, (struct sockaddr *) &serv_addr, sizeof(serv_addr)) < 0) {
                throw std::runtime_error("Bind failed");
            }

            if (listen(sockfd, 1) < 0) {
                throw std::runtime_error("Listen failed");
            }

            std::cout << "Worker listening on port " << port << "..." << std::endl;

            socklen_t clilen = sizeof(serv_addr);
            ingress_fd = accept(sockfd, (struct sockaddr *) &serv_addr, &clilen);
            if (ingress_fd < 0) {
                throw std::runtime_error("Accept failed");
            }

            setup_socket_low_latency(ingress_fd);
            std::cout << "Previous node connected!" << std::endl;

            // 2. Connect to Next (Egress) - Optional if not Tail
            if (!next_ip.empty()) {
                egress_fd = socket(AF_INET, SOCK_STREAM, 0);
                if (egress_fd < 0) throw std::runtime_error("Failed to create egress socket");

                sockaddr_in next_addr{};
                next_addr.sin_family = AF_INET;
                next_addr.sin_port = htons(next_port);
                if (inet_pton(AF_INET, next_ip.c_str(), &next_addr.sin_addr) <= 0) {
                    throw std::runtime_error("Invalid next address");
                }

                std::cout << "Connecting to next worker at " << next_ip << ":" << next_port << "..." << std::endl;
                // Simple retry loop for connecting to next worker
                int retries = 10;
                while (connect(egress_fd, (struct sockaddr *) &next_addr, sizeof(next_addr)) < 0) {
                    if (--retries == 0) throw std::runtime_error("Connection to next worker failed");
                    std::cout << "Waiting for next worker..." << std::endl;
                    sleep(1);
                }

                setup_socket_low_latency(egress_fd);
                std::cout << "Connected to next worker!" << std::endl;
            }
        } else {
            // Client Mode (Master)
            // Connects to Worker (treated as Next Node)
            egress_fd = sockfd; // Use sockfd as egress
            
            sockaddr_in serv_addr{};
            serv_addr.sin_family = AF_INET;
            serv_addr.sin_port = htons(port);
            if (inet_pton(AF_INET, ip.c_str(), &serv_addr.sin_addr) <= 0) {
                throw std::runtime_error("Invalid address/ Address not supported");
            }

            std::cout << "Connecting to worker at " << ip << ":" << port << "..." << std::endl;
            if (connect(egress_fd, (struct sockaddr *) &serv_addr, sizeof(serv_addr)) < 0) {
                throw std::runtime_error("Connection failed");
            }

            setup_socket_low_latency(egress_fd);
            std::cout << "Connected to worker!" << std::endl;
        }
    }

    void TcpTransport::send(const void *data, const size_t size) {
        if (is_server) {
            send_prev(data, size); // Worker sends to master/prev
        } else {
            send_next(data, size); // Master sends to worker/next
        }
    }

    void TcpTransport::recv(void *data, const size_t size) {
        if (is_server) {
            recv_prev(data, size); // Worker recvs from master/prev
        } else {
            recv_next(data, size); // Master recvs from worker/next
        }
    }

    void TcpTransport::send_next(const void *data, size_t size) {
        if (egress_fd == -1) throw std::runtime_error("No next node connected");
        size_t total_sent = 0;
        const auto *bytes = static_cast<const char *>(data);
        while (total_sent < size) {
            ssize_t sent = ::send(egress_fd, bytes + total_sent, size - total_sent, 0);
            if (sent < 0) throw std::runtime_error("Send next failed");
            total_sent += sent;
        }
    }

    void TcpTransport::recv_next(void *data, size_t size) {
        if (egress_fd == -1) throw std::runtime_error("No next node connected");
        size_t total_received = 0;
        auto *bytes = static_cast<char *>(data);
        while (total_received < size) {
            ssize_t received = ::recv(egress_fd, bytes + total_received, size - total_received, 0);
            if (received < 0) throw std::runtime_error("Recv next failed");
            if (received == 0) throw std::runtime_error("Next node disconnected");
            total_received += received;
        }
    }

    void TcpTransport::send_prev(const void *data, size_t size) {
        if (ingress_fd == -1) throw std::runtime_error("No prev node connected");
        size_t total_sent = 0;
        const auto *bytes = static_cast<const char *>(data);
        while (total_sent < size) {
            ssize_t sent = ::send(ingress_fd, bytes + total_sent, size - total_sent, 0);
            if (sent < 0) throw std::runtime_error("Send prev failed");
            total_sent += sent;
        }
    }

    void TcpTransport::recv_prev(void *data, size_t size) {
        if (ingress_fd == -1) throw std::runtime_error("No prev node connected");
        size_t total_received = 0;
        auto *bytes = static_cast<char *>(data);
        while (total_received < size) {
            ssize_t received = ::recv(ingress_fd, bytes + total_received, size - total_received, 0);
            if (received < 0) throw std::runtime_error("Recv prev failed");
            if (received == 0) throw std::runtime_error("Prev node disconnected");
            total_received += received;
        }
    }
} // namespace Inference