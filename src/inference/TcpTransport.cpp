#include "TcpTransport.h"
#include <sys/socket.h>
#include <arpa/inet.h>
#include <unistd.h>
#include <netinet/tcp.h>
#include <stdexcept>
#include <iostream>
#include <cstring>
#include <thread>
#include <chrono>

namespace Inference {
    TcpTransport::TcpTransport(std::string ip, const int port, const bool is_server, std::string next_ip, int next_port)
        : ip(std::move(ip)), port(port), is_server(is_server), next_ip(std::move(next_ip)), next_port(next_port) {
    }

    TcpTransport::~TcpTransport() {
        if (connfd != -1) close(connfd);
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
        // Shared Setup: Create Listener Socket (Input) if Server OR Ring Mode
        // In Ring Mode, every node listens for its Predecessor.
        if (is_server || !next_ip.empty()) {
            sockfd = socket(AF_INET, SOCK_STREAM, 0);
            if (sockfd < 0) throw std::runtime_error("Failed to create listener socket");

            int opt = 1;
            if (setsockopt(sockfd, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt)) < 0) {
                throw std::runtime_error("setsockopt failed");
            }

            sockaddr_in serv_addr{};
            serv_addr.sin_family = AF_INET;
            serv_addr.sin_addr.s_addr = INADDR_ANY;
            serv_addr.sin_port = htons(port);

            if (bind(sockfd, (struct sockaddr *) &serv_addr, sizeof(serv_addr)) < 0) {
                throw std::runtime_error("Bind failed. Is the port already in use?");
            }

            if (listen(sockfd, 10) < 0) { // Backlog 10 to be safe
                throw std::runtime_error("Listen failed");
            }

            std::cout << "Node listening on port " << port << "..." << std::endl;
        }

        // Ring Mode / Output Setup
        if (!next_ip.empty()) {
            // We need to connect to Next Node
            std::cout << "Ring Mode: Connecting to next node at " << next_ip << ":" << next_port << "..." << std::endl;
            
            connfd = socket(AF_INET, SOCK_STREAM, 0);
            if (connfd < 0) throw std::runtime_error("Failed to create output socket");

            sockaddr_in next_addr{};
            next_addr.sin_family = AF_INET;
            next_addr.sin_port = htons(next_port);
            if (inet_pton(AF_INET, next_ip.c_str(), &next_addr.sin_addr) <= 0) {
                throw std::runtime_error("Invalid next_ip address");
            }

            // Retry logic for connection (Wait for Next Node to start listening)
            int retries = 0;
            while (true) {
                if (connect(connfd, (struct sockaddr *) &next_addr, sizeof(next_addr)) == 0) {
                    break;
                }
                std::cout << "Waiting for next node (" << next_ip << ":" << next_port << ")..." << std::endl;
                std::this_thread::sleep_for(std::chrono::seconds(1));
                retries++;
                if (retries > 60) throw std::runtime_error("Connection timed out");
            }

            setup_socket_low_latency(connfd);
            std::cout << "Connected to next node!" << std::endl;
        } else if (!is_server) {
            // Legacy Client Mode (Connect to Worker)
            sockfd = socket(AF_INET, SOCK_STREAM, 0);
            if (sockfd < 0) throw std::runtime_error("Failed to create socket");

            sockaddr_in serv_addr{};
            serv_addr.sin_family = AF_INET;
            serv_addr.sin_port = htons(port);
            if (inet_pton(AF_INET, ip.c_str(), &serv_addr.sin_addr) <= 0) {
                throw std::runtime_error("Invalid address");
            }

            std::cout << "Legacy: Connecting to worker at " << ip << ":" << port << "..." << std::endl;
            if (connect(sockfd, (struct sockaddr *) &serv_addr, sizeof(serv_addr)) < 0) {
                throw std::runtime_error("Connection failed");
            }
            setup_socket_low_latency(sockfd);
            std::cout << "Connected to worker!" << std::endl;
        }

        // Finalize Input: Accept Connection
        if (is_server || !next_ip.empty()) {
            sockaddr_in client_addr{};
            socklen_t clilen = sizeof(client_addr);
            
            std::cout << "Waiting for incoming connection..." << std::endl;
            clientfd = accept(sockfd, (struct sockaddr *) &client_addr, &clilen);
            if (clientfd < 0) {
                throw std::runtime_error("Accept failed");
            }

            setup_socket_low_latency(clientfd);
            std::cout << "Incoming connection accepted from " << inet_ntoa(client_addr.sin_addr) << std::endl;
        }
    }

    void TcpTransport::send(const void *data, const size_t size) {
        // Ring Mode: Send to connfd (Next)
        // Legacy Server: Send to clientfd
        // Legacy Client: Send to sockfd
        int fd;
        if (!next_ip.empty()) fd = connfd;
        else fd = is_server ? clientfd : sockfd;

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
        // Ring Mode: Recv from clientfd (Prev)
        // Legacy Server: Recv from clientfd
        // Legacy Client: Recv from sockfd
        int fd;
        if (!next_ip.empty()) fd = clientfd;
        else fd = is_server ? clientfd : sockfd;

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