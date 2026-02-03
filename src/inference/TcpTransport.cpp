#include "TcpTransport.h"
#include <sys/socket.h>
#include <arpa/inet.h>
#include <unistd.h>
#include <netinet/tcp.h>
#include <stdexcept>
#include <iostream>
#include <cstring>
#include <sys/uio.h> // For writev

namespace Inference {
    TcpTransport::TcpTransport(std::string ip, const int port, std::string next_ip, int next_port)
        : ip(std::move(ip)), port(port), next_ip(std::move(next_ip)), next_port(next_port) {
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
        // 1. Bind and Listen (Ingress) - Everyone listens in the Ring
        sockfd = socket(AF_INET, SOCK_STREAM, 0);
        if (sockfd < 0) {
            throw std::runtime_error("Failed to create socket");
        }

        // Optimization: Increase Socket Buffers to 8MB
        int buf_size = 8 * 1024 * 1024;
        setsockopt(sockfd, SOL_SOCKET, SO_RCVBUF, &buf_size, sizeof(buf_size));
        setsockopt(sockfd, SOL_SOCKET, SO_SNDBUF, &buf_size, sizeof(buf_size));

        sockaddr_in serv_addr{};
        serv_addr.sin_family = AF_INET;
        serv_addr.sin_port = htons(port);
        serv_addr.sin_addr.s_addr = INADDR_ANY; // Bind to all interfaces

        int opt = 1;
        if (setsockopt(sockfd, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt)) < 0) {
            throw std::runtime_error("setsockopt failed");
        }

        if (bind(sockfd, (struct sockaddr *) &serv_addr, sizeof(serv_addr)) < 0) {
            throw std::runtime_error("Bind failed on port " + std::to_string(port));
        }

        if (listen(sockfd, 1) < 0) {
            throw std::runtime_error("Listen failed");
        }

        std::cout << "Node listening on port " << port << "..." << std::endl;

        // 2. Connect to Next (Egress)
        // Must have a next node in a ring
        if (next_ip.empty()) {
             throw std::runtime_error("Next IP is required for Ring Topology");
        }

        egress_fd = socket(AF_INET, SOCK_STREAM, 0);
        if (egress_fd < 0) throw std::runtime_error("Failed to create egress socket");

        sockaddr_in next_addr{};
        next_addr.sin_family = AF_INET;
        next_addr.sin_port = htons(next_port);
        if (inet_pton(AF_INET, next_ip.c_str(), &next_addr.sin_addr) <= 0) {
            throw std::runtime_error("Invalid next address");
        }

        std::cout << "Connecting to next node at " << next_ip << ":" << next_port << "..." << std::endl;
        
        // Retry loop
        int retries = 100; // More retries for ring stability
        int wait_time = 1;
        while (connect(egress_fd, (struct sockaddr *) &next_addr, sizeof(next_addr)) < 0) {
            if (--retries == 0) throw std::runtime_error("Connection to next node failed.");
            
            std::cout << "Waiting for next node... (retrying in " << wait_time << "s)" << std::endl;
            sleep(wait_time);
            
            if (wait_time < 5) wait_time *= 2;
        }

        setup_socket_low_latency(egress_fd);
        std::cout << "Connected to next node!" << std::endl;

        // 3. Accept Previous (Ingress)
        socklen_t clilen = sizeof(serv_addr);
        std::cout << "Waiting for previous node to connect..." << std::endl;
        ingress_fd = accept(sockfd, (struct sockaddr *) &serv_addr, &clilen);
        if (ingress_fd < 0) {
            throw std::runtime_error("Accept failed");
        }

        setup_socket_low_latency(ingress_fd);
        std::cout << "Previous node connected!" << std::endl;
    }

    void TcpTransport::send(const void *data, const size_t size) {
        send_next(data, size);
    }

    void TcpTransport::recv(void *data, const size_t size) {
        recv_prev(data, size);
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

    void TcpTransport::send_multipart_next(const void* header, size_t header_size, const void* data, size_t data_size) {
        if (egress_fd == -1) throw std::runtime_error("No next node connected");
        
        struct iovec iov[2];
        iov[0].iov_base = const_cast<void*>(header);
        iov[0].iov_len = header_size;
        iov[1].iov_base = const_cast<void*>(data);
        iov[1].iov_len = data_size;

        // writev sends atomic list
        ssize_t sent = writev(egress_fd, iov, 2);
        
        if (sent < 0) throw std::runtime_error("Writev failed");
        
        // Handle partial sends (rare with large buffers but possible)
        size_t total_expected = header_size + data_size;
        if (static_cast<size_t>(sent) < total_expected) {
            // Fallback to sending the rest sequentially if partial write happened
            // This is complex logic, simplified here by assuming buffers > packet
            size_t remaining = total_expected - sent;
            
            // Revert to manual send for tail
            // Calculate where we left off
            size_t sent_so_far = sent;
            if (sent_so_far < header_size) {
                // Sent part of header
                send_next((char*)header + sent_so_far, header_size - sent_so_far);
                send_next(data, data_size);
            } else {
                // Sent all header + part of data
                size_t data_sent = sent_so_far - header_size;
                send_next((char*)data + data_sent, data_size - data_sent);
            }
        }
    }

    void TcpTransport::recv_next(void *data, size_t size) {
        // Not used in Ring Forward-only flow, but kept for interface compliance or debug
        throw std::runtime_error("recv_next not supported in O(1) Ring Topology");
    }

    void TcpTransport::send_prev(const void *data, size_t size) {
         // Not used in Ring Forward-only flow
         throw std::runtime_error("send_prev not supported in O(1) Ring Topology");
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