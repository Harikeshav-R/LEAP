#include "TcpTransport.h"
#include <sys/socket.h>
#include <arpa/inet.h>
#include <unistd.h>
#include <netinet/tcp.h>
#include <stdexcept>
#include <iostream>
#include <cstring>

namespace Inference {
    TcpTransport::TcpTransport(std::string ip, const int port, const bool is_server)
        : ip(std::move(ip)), port(port), is_server(is_server) {
    }

    TcpTransport::~TcpTransport() {
        if (clientfd != -1) close(clientfd);
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

        sockaddr_in serv_addr{};
        serv_addr.sin_family = AF_INET;
        serv_addr.sin_port = htons(port);

        if (is_server) {
            // Server Mode (Worker)
            serv_addr.sin_addr.s_addr = INADDR_ANY;

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
            clientfd = accept(sockfd, (struct sockaddr *) &serv_addr, &clilen);
            if (clientfd < 0) {
                throw std::runtime_error("Accept failed");
            }

            setup_socket_low_latency(clientfd);
            std::cout << "Master connected!" << std::endl;
        } else {
            // Client Mode (Master)
            if (inet_pton(AF_INET, ip.c_str(), &serv_addr.sin_addr) <= 0) {
                throw std::runtime_error("Invalid address/ Address not supported");
            }

            std::cout << "Connecting to worker at " << ip << ":" << port << "..." << std::endl;
            if (connect(sockfd, (struct sockaddr *) &serv_addr, sizeof(serv_addr)) < 0) {
                throw std::runtime_error("Connection failed");
            }

            setup_socket_low_latency(sockfd);
            std::cout << "Connected to worker!" << std::endl;
        }
    }

    void TcpTransport::send(const void *data, const size_t size) {
        const int fd = is_server ? clientfd : sockfd;
        size_t total_sent = 0;
        const auto *bytes = static_cast<const char *>(data);

        while (total_sent < size) {
            const ssize_t sent = ::send(fd, bytes + total_sent, size - total_sent, 0);
            if (sent < 0) {
                throw std::runtime_error("Send failed");
            }
            total_sent += sent;
        }
    }

    void TcpTransport::recv(void *data, const size_t size) {
        const int fd = is_server ? clientfd : sockfd;
        size_t total_received = 0;
        auto *bytes = static_cast<char *>(data);

        while (total_received < size) {
            const ssize_t received = ::recv(fd, bytes + total_received, size - total_received, 0);
            if (received < 0) {
                throw std::runtime_error("Receive failed");
            }
            if (received == 0) {
                throw std::runtime_error("Connection closed by peer");
            }
            total_received += received;
        }
    }
} // namespace Inference