#include "UdpTransport.h"
#include "../kernel/leap_protocol.h"
#include <sys/socket.h>
#include <arpa/inet.h>
#include <unistd.h>
#include <stdexcept>
#include <iostream>
#include <cstring>
#include <algorithm>

namespace Inference {
    UdpTransport::UdpTransport(std::string ip, const int port, std::string next_ip, int next_port)
        : ip(std::move(ip)), port(port), next_ip(std::move(next_ip)), next_port(next_port) {
        reassembly_buffer.resize(LEAP_BUFFER_SIZE);
        packet_buffer.resize(2048);
        memset(&prev_addr, 0, sizeof(prev_addr));
        memset(&next_addr, 0, sizeof(next_addr));
    }

    UdpTransport::~UdpTransport() {
        if (sockfd != -1) close(sockfd);
    }

    void UdpTransport::initialize() {
        sockfd = socket(AF_INET, SOCK_DGRAM, 0);
        if (sockfd < 0) throw std::runtime_error("Socket creation failed");

        // Bind to local port
        sockaddr_in bind_addr{};
        bind_addr.sin_family = AF_INET;
        bind_addr.sin_port = htons(port);
        bind_addr.sin_addr.s_addr = INADDR_ANY;

        // Optimization: Increase Socket Buffers to 8MB to absorb bursts
        int buf_size = 8 * 1024 * 1024;
        setsockopt(sockfd, SOL_SOCKET, SO_RCVBUF, &buf_size, sizeof(buf_size));
        setsockopt(sockfd, SOL_SOCKET, SO_SNDBUF, &buf_size, sizeof(buf_size));

        if (bind(sockfd, (struct sockaddr *) &bind_addr, sizeof(bind_addr)) < 0) {
            throw std::runtime_error("Bind failed on port " + std::to_string(port));
        }

        std::cout << "UDP Node bound to port " << port << std::endl;

        // Configure Next Addr
        if (next_ip.empty()) {
            throw std::runtime_error("Next IP required for Ring");
        }
        next_addr.sin_family = AF_INET;
        next_addr.sin_port = htons(next_port);
        if (inet_pton(AF_INET, next_ip.c_str(), &next_addr.sin_addr) <= 0) {
            throw std::runtime_error("Invalid Next IP");
        }
        std::cout << "UDP Chain: Next node at " << next_ip << ":" << next_port << std::endl;
    }

    void UdpTransport::send(const void *data, const size_t size) {
        send_next(data, size);
    }

    void UdpTransport::recv(void *data, const size_t size) {
        recv_prev(data, size);
    }

    void UdpTransport::send_next(const void *data, size_t size) {
        if (next_addr.sin_family == 0) throw std::runtime_error("Next address not configured");
        
        const auto *bytes = static_cast<const uint8_t *>(data);
        size_t processed = 0;
        uint16_t total_chunks = (size + LEAP_CHUNK_SIZE - 1) / LEAP_CHUNK_SIZE;
        uint16_t chunk_idx = 0;

        seq_id++;

        // Reuse packet_buffer for HEADER only (saves large allocs, keeps alignment)
        if (packet_buffer.size() < sizeof(leap_header)) {
             packet_buffer.resize(sizeof(leap_header));
        }
        auto *hdr = reinterpret_cast<leap_header *>(packet_buffer.data());

        while (processed < size) {
            size_t chunk_len = LEAP_CHUNK_SIZE;
            if (processed + chunk_len > size) chunk_len = size - processed;

            // Prepare Header
            hdr->magic = htonl(LEAP_MAGIC);
            hdr->seq_id = htons(seq_id);
            hdr->chunk_id = htons(chunk_idx);
            hdr->total_chunks = htons(total_chunks);

            // Optimization: Scatter/Gather I/O (Zero-Copy)
            // vector 0: Header
            // vector 1: Payload (Direct pointer to data)
            struct iovec iov[2];
            iov[0].iov_base = packet_buffer.data();
            iov[0].iov_len = sizeof(leap_header);
            iov[1].iov_base = const_cast<uint8_t*>(bytes + processed);
            iov[1].iov_len = chunk_len;

            struct msghdr msg{};
            msg.msg_name = &next_addr;
            msg.msg_namelen = sizeof(next_addr);
            msg.msg_iov = iov;
            msg.msg_iovlen = 2;

            if (sendmsg(sockfd, &msg, 0) < 0) {
                throw std::runtime_error("UDP send next failed");
            }

            processed += chunk_len;
            chunk_idx++;
        }
    }

    void UdpTransport::recv_next(void *data, size_t size) {
         throw std::runtime_error("recv_next not supported in Ring");
    }

    void UdpTransport::send_prev(const void *data, size_t size) {
        throw std::runtime_error("send_prev not supported in Ring");
    }

    void UdpTransport::recv_prev(void *data, size_t size) {
        size_t expected_chunks = (size + LEAP_CHUNK_SIZE - 1) / LEAP_CHUNK_SIZE;
        std::vector<bool> received_chunks(expected_chunks, false);
        size_t chunks_count = 0;
        auto *out_bytes = static_cast<uint8_t *>(data);
        int32_t active_seq_id = -1;
        // reuse member buffer packet_buffer

        while (chunks_count < expected_chunks) {
            sockaddr_in src_addr;
            socklen_t addr_len = sizeof(src_addr);

            ssize_t len = recvfrom(sockfd, packet_buffer.data(), packet_buffer.size(), 0,
                                   (struct sockaddr *) &src_addr, &addr_len);

            if (len < 0) throw std::runtime_error("UDP recv prev failed");
            
            // Learn Prev Addr (optional, just logging or verification)
            if (!prev_addr_set) {
                prev_addr = src_addr;
                prev_addr_set = true;
            }

            if (static_cast<size_t>(len) < sizeof(leap_header)) continue;

            auto *hdr = reinterpret_cast<leap_header *>(packet_buffer.data());
            if (ntohl(hdr->magic) != LEAP_MAGIC) continue;

            uint16_t seq = ntohs(hdr->seq_id);
            uint16_t chunk = ntohs(hdr->chunk_id);

            if (active_seq_id == -1) {
                active_seq_id = seq;
            } else if (seq != active_seq_id) {
                continue;
            }

            if (chunk >= expected_chunks) continue;
            if (received_chunks[chunk]) continue;

            size_t payload_len = len - sizeof(leap_header);
            size_t offset = chunk * LEAP_CHUNK_SIZE;

            if (offset + payload_len > size) payload_len = size - offset;

            std::memcpy(out_bytes + offset, packet_buffer.data() + sizeof(leap_header), payload_len);
            received_chunks[chunk] = true;
            chunks_count++;
        }
    }
}