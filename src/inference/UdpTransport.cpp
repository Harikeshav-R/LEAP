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
    UdpTransport::UdpTransport(std::string ip, const int port, const bool is_server)
        : ip(std::move(ip)), port(port), is_server(is_server) {
        reassembly_buffer.resize(LEAP_BUFFER_SIZE);
    }

    UdpTransport::~UdpTransport() {
        if (sockfd != -1) close(sockfd);
    }

    void UdpTransport::initialize() {
        sockfd = socket(AF_INET, SOCK_DGRAM, 0);
        if (sockfd < 0) throw std::runtime_error("Socket creation failed");

        memset(&dest_addr, 0, sizeof(dest_addr));
        dest_addr.sin_family = AF_INET;
        dest_addr.sin_port = htons(port);

        if (is_server) {
            dest_addr.sin_addr.s_addr = INADDR_ANY;
            if (bind(sockfd, (struct sockaddr *) &dest_addr, sizeof(dest_addr)) < 0) {
                throw std::runtime_error("Bind failed");
            }
            std::cout << "UDP Worker listening on port " << port << "..." << std::endl;
        } else {
            if (inet_pton(AF_INET, ip.c_str(), &dest_addr.sin_addr) <= 0) {
                throw std::runtime_error("Invalid IP");
            }
            std::cout << "UDP Master ready. Sending to " << ip << ":" << port << std::endl;
        }
    }

    void UdpTransport::send(const void *data, const size_t size) {
        const auto *bytes = static_cast<const uint8_t *>(data);
        size_t processed = 0;
        uint16_t total_chunks = (size + LEAP_CHUNK_SIZE - 1) / LEAP_CHUNK_SIZE;
        uint16_t chunk_idx = 0;

        seq_id++;

        while (processed < size) {
            size_t chunk_len = LEAP_CHUNK_SIZE;
            if (processed + chunk_len > size) chunk_len = size - processed;

            // Construct Packet
            std::vector<uint8_t> packet(sizeof(leap_header) + chunk_len);
            auto *hdr = reinterpret_cast<leap_header *>(packet.data());

            hdr->magic = htonl(LEAP_MAGIC);
            hdr->seq_id = htons(seq_id);
            hdr->chunk_id = htons(chunk_idx);
            hdr->total_chunks = htons(total_chunks);

            std::memcpy(packet.data() + sizeof(leap_header), bytes + processed, chunk_len);

            if (sendto(sockfd, packet.data(), packet.size(), 0,
                       (struct sockaddr *) &dest_addr, sizeof(dest_addr)) < 0) {
                throw std::runtime_error("UDP send failed");
            }

            processed += chunk_len;
            chunk_idx++;
        }
    }

    void UdpTransport::recv(void *data, const size_t size) {
        // Naive reassembly: assumes packets arrive in order and no loss (LAN assumption)
        // A robust implementation would use a map/buffer to handle out-of-order packets.
        size_t received_bytes = 0;
        uint8_t expected_chunk = 0;
        auto *out_bytes = static_cast<uint8_t *>(data);

        while (received_bytes < size) {
            std::vector<uint8_t> packet(2048); // MTU safety
            sockaddr_in src_addr;
            socklen_t addr_len = sizeof(src_addr);

            ssize_t len = recvfrom(sockfd, packet.data(), packet.size(), 0,
                                   (struct sockaddr *) &src_addr, &addr_len);

            if (len < 0) throw std::runtime_error("UDP recv failed");

            // If we are a worker (server), remember who sent us data so we can reply
            if (is_server &&dest_addr
            .
            sin_addr.s_addr == INADDR_ANY
            )
            {
                dest_addr = src_addr;
                std::cout << "Worker: Locked onto Master at " << inet_ntoa(src_addr.sin_addr)
                        << ":" << ntohs(src_addr.sin_port) << std::endl;
            }

            if (static_cast<size_t>(len) < sizeof(leap_header)) continue;

            auto *hdr = reinterpret_cast<leap_header *>(packet.data());
            if (ntohl(hdr->magic) != LEAP_MAGIC) continue; // Not our packet

            size_t payload_len = len - sizeof(leap_header);

            // In a real implementation, check seq_id and chunk_id
            // Here, we just copy blindly for minimal latency demonstration
            if (received_bytes + payload_len > size) payload_len = size - received_bytes;

            std::memcpy(out_bytes + received_bytes, packet.data() + sizeof(leap_header), payload_len);
            received_bytes += payload_len;
        }
    }
}