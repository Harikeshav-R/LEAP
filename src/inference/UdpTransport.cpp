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
    UdpTransport::UdpTransport(std::string ip, const int port, const bool is_server, std::string next_ip, int next_port)
        : ip(std::move(ip)), port(port), is_server(is_server), next_ip(std::move(next_ip)), next_port(next_port) {
        reassembly_buffer.resize(LEAP_BUFFER_SIZE);
        memset(&prev_addr, 0, sizeof(prev_addr));
        memset(&next_addr, 0, sizeof(next_addr));
    }

    UdpTransport::~UdpTransport() {
        if (sockfd != -1) close(sockfd);
    }

    void UdpTransport::initialize() {
        sockfd = socket(AF_INET, SOCK_DGRAM, 0);
        if (sockfd < 0) throw std::runtime_error("Socket creation failed");

        // Always bind to local port (for both Master and Worker) to ensure symmetric addressing
        // This is critical when talking to KernelTransport which expects replies on the same port
        sockaddr_in bind_addr{};
        bind_addr.sin_family = AF_INET;
        bind_addr.sin_port = htons(port);
        bind_addr.sin_addr.s_addr = INADDR_ANY;
        if (bind(sockfd, (struct sockaddr *) &bind_addr, sizeof(bind_addr)) < 0) {
            throw std::runtime_error("Bind failed on port " + std::to_string(port));
        }

        if (is_server) {
            std::cout << "UDP Worker listening on port " << port << "..." << std::endl;
        } else {
            std::cout << "UDP Master bound to port " << port << std::endl;
        }

        // Configure Next Addr
        if (!next_ip.empty()) {
            next_addr.sin_family = AF_INET;
            next_addr.sin_port = htons(next_port);
            if (inet_pton(AF_INET, next_ip.c_str(), &next_addr.sin_addr) <= 0) {
                throw std::runtime_error("Invalid Next IP");
            }
            std::cout << "UDP Chain: Next node at " << next_ip << ":" << next_port << std::endl;
        }
    }

    void UdpTransport::send(const void *data, const size_t size) {
        if (is_server) send_prev(data, size);
        else send_next(data, size);
    }

    void UdpTransport::recv(void *data, const size_t size) {
        if (is_server) recv_prev(data, size);
        else recv_next(data, size);
    }

    void UdpTransport::send_next(const void *data, size_t size) {
        if (next_addr.sin_family == 0) throw std::runtime_error("Next address not configured");
        
        const auto *bytes = static_cast<const uint8_t *>(data);
        size_t processed = 0;
        uint16_t total_chunks = (size + LEAP_CHUNK_SIZE - 1) / LEAP_CHUNK_SIZE;
        uint16_t chunk_idx = 0;

        seq_id++;

        while (processed < size) {
            size_t chunk_len = LEAP_CHUNK_SIZE;
            if (processed + chunk_len > size) chunk_len = size - processed;

            std::vector<uint8_t> packet(sizeof(leap_header) + chunk_len);
            auto *hdr = reinterpret_cast<leap_header *>(packet.data());

            hdr->magic = htonl(LEAP_MAGIC);
            hdr->seq_id = htons(seq_id);
            hdr->chunk_id = htons(chunk_idx);
            hdr->total_chunks = htons(total_chunks);

            std::memcpy(packet.data() + sizeof(leap_header), bytes + processed, chunk_len);

            if (sendto(sockfd, packet.data(), packet.size(), 0,
                       (struct sockaddr *) &next_addr, sizeof(next_addr)) < 0) {
                throw std::runtime_error("UDP send next failed");
            }

            processed += chunk_len;
            chunk_idx++;
        }
    }

    void UdpTransport::recv_next(void *data, size_t size) {
        size_t expected_chunks = (size + LEAP_CHUNK_SIZE - 1) / LEAP_CHUNK_SIZE;
        std::vector<bool> received_chunks(expected_chunks, false);
        size_t chunks_count = 0;
        auto *out_bytes = static_cast<uint8_t *>(data);
        int32_t active_seq_id = -1;
        std::vector<uint8_t> packet(2048);

        while (chunks_count < expected_chunks) {
            sockaddr_in src_addr;
            socklen_t addr_len = sizeof(src_addr);

            ssize_t len = recvfrom(sockfd, packet.data(), packet.size(), 0,
                                   (struct sockaddr *) &src_addr, &addr_len);

            if (len < 0) throw std::runtime_error("UDP recv next failed");
            if (static_cast<size_t>(len) < sizeof(leap_header)) continue;

            auto *hdr = reinterpret_cast<leap_header *>(packet.data());
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

            std::memcpy(out_bytes + offset, packet.data() + sizeof(leap_header), payload_len);
            received_chunks[chunk] = true;
            chunks_count++;
        }
    }

    void UdpTransport::send_prev(const void *data, size_t size) {
        if (!prev_addr_set) throw std::runtime_error("Previous address not known");

        const auto *bytes = static_cast<const uint8_t *>(data);
        size_t processed = 0;
        uint16_t total_chunks = (size + LEAP_CHUNK_SIZE - 1) / LEAP_CHUNK_SIZE;
        uint16_t chunk_idx = 0;

        seq_id++;

        while (processed < size) {
            size_t chunk_len = LEAP_CHUNK_SIZE;
            if (processed + chunk_len > size) chunk_len = size - processed;

            std::vector<uint8_t> packet(sizeof(leap_header) + chunk_len);
            auto *hdr = reinterpret_cast<leap_header *>(packet.data());

            hdr->magic = htonl(LEAP_MAGIC);
            hdr->seq_id = htons(seq_id);
            hdr->chunk_id = htons(chunk_idx);
            hdr->total_chunks = htons(total_chunks);

            std::memcpy(packet.data() + sizeof(leap_header), bytes + processed, chunk_len);

            if (sendto(sockfd, packet.data(), packet.size(), 0,
                       (struct sockaddr *) &prev_addr, sizeof(prev_addr)) < 0) {
                throw std::runtime_error("UDP send prev failed");
            }

            processed += chunk_len;
            chunk_idx++;
        }
    }

    void UdpTransport::recv_prev(void *data, size_t size) {
        size_t expected_chunks = (size + LEAP_CHUNK_SIZE - 1) / LEAP_CHUNK_SIZE;
        std::vector<bool> received_chunks(expected_chunks, false);
        size_t chunks_count = 0;
        auto *out_bytes = static_cast<uint8_t *>(data);
        int32_t active_seq_id = -1;
        std::vector<uint8_t> packet(2048);

        while (chunks_count < expected_chunks) {
            sockaddr_in src_addr;
            socklen_t addr_len = sizeof(src_addr);

            ssize_t len = recvfrom(sockfd, packet.data(), packet.size(), 0,
                                   (struct sockaddr *) &src_addr, &addr_len);

            if (len < 0) throw std::runtime_error("UDP recv prev failed");
            
            // Learn Prev Addr
            if (!prev_addr_set) {
                prev_addr = src_addr;
                prev_addr_set = true;
                std::cout << "UDP locked onto Prev at " << inet_ntoa(src_addr.sin_addr) << ":" << ntohs(src_addr.sin_port) << std::endl;
            }

            if (static_cast<size_t>(len) < sizeof(leap_header)) continue;

            auto *hdr = reinterpret_cast<leap_header *>(packet.data());
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

            std::memcpy(out_bytes + offset, packet.data() + sizeof(leap_header), payload_len);
            received_chunks[chunk] = true;
            chunks_count++;
        }
    }
}