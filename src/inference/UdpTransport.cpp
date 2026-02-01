#include "UdpTransport.h"
#include "../kernel/leap_protocol.h"
#include <sys/socket.h>
#include <arpa/inet.h>
#include <unistd.h>
#include <stdexcept>
#include <iostream>
#include <cstring>
#include <algorithm>
#include <cmath>

namespace Inference {
    UdpTransport::UdpTransport(std::string ip, const int port, const bool is_server)
        : ip(std::move(ip)), port(port), is_server(is_server) {
        // Reserve 8MB for the internal reassembly buffer
        rx_buffer.resize(LEAP_BUFFER_SIZE);
        // Initial bitmap size (will grow if needed, but starting with a reasonable guess helps)
        rx_chunk_bitmap.resize(1024, false);
    }

    UdpTransport::~UdpTransport() {
        if (sockfd != -1) close(sockfd);
    }

    void UdpTransport::initialize() {
        sockfd = socket(AF_INET, SOCK_DGRAM, 0);
        if (sockfd < 0) throw std::runtime_error("Socket creation failed");

        // Set buffer sizes to handle bursts
        int rcvbuf = 4 * 1024 * 1024; // 4MB OS Recv Buffer
        setsockopt(sockfd, SOL_SOCKET, SO_RCVBUF, &rcvbuf, sizeof(rcvbuf));

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

        tx_seq_id++;

        // Reuse a single packet buffer to minimize allocation overhead
        std::vector<uint8_t> packet(sizeof(leap_header) + LEAP_CHUNK_SIZE);
        auto *hdr = reinterpret_cast<leap_header *>(packet.data());

        while (processed < size) {
            size_t chunk_len = LEAP_CHUNK_SIZE;
            if (processed + chunk_len > size) chunk_len = size - processed;

            hdr->magic = htonl(LEAP_MAGIC);
            hdr->seq_id = htons(tx_seq_id);
            hdr->chunk_id = htons(chunk_idx);
            hdr->total_chunks = htons(total_chunks);

            std::memcpy(packet.data() + sizeof(leap_header), bytes + processed, chunk_len);

            // Send with effective size
            if (sendto(sockfd, packet.data(), sizeof(leap_header) + chunk_len, 0,
                       (struct sockaddr *) &dest_addr, sizeof(dest_addr)) < 0) {
                throw std::runtime_error("UDP send failed");
            }

            processed += chunk_len;
            chunk_idx++;
        }
    }

    void UdpTransport::recv(void *data, const size_t size) {
        if (size > rx_buffer.size()) {
            throw std::runtime_error("Receive size exceeds internal buffer size");
        }

        uint8_t temp_packet[2048]; // Stack buffer for incoming packet
        sockaddr_in src_addr;
        socklen_t addr_len = sizeof(src_addr);

        while (true) {
            ssize_t len = recvfrom(sockfd, temp_packet, sizeof(temp_packet), 0,
                                   (struct sockaddr *) &src_addr, &addr_len);

            if (len < 0) throw std::runtime_error("UDP recv failed");
            if (static_cast<size_t>(len) < sizeof(leap_header)) continue;

            auto *hdr = reinterpret_cast<leap_header *>(temp_packet);
            if (ntohl(hdr->magic) != LEAP_MAGIC) continue;

            uint16_t seq = ntohs(hdr->seq_id);
            uint16_t chunk_id = ntohs(hdr->chunk_id);
            uint16_t total_chunks = ntohs(hdr->total_chunks);
            size_t payload_len = len - sizeof(leap_header);

            // Worker auto-lock logic
            if (is_server && dest_addr.sin_addr.s_addr == INADDR_ANY) {
                dest_addr = src_addr;
                std::cout << "Worker: Locked onto Master at " << inet_ntoa(src_addr.sin_addr)
                          << ":" << ntohs(src_addr.sin_port) << std::endl;
            }

            // Sequence Logic
            int16_t diff = static_cast<int16_t>(seq - rx_active_seq_id);

            if (diff > 0) {
                // NEW FRAME DETECTED
                rx_active_seq_id = seq;
                rx_chunks_received = 0;
                
                // Resize bitmap if total_chunks is larger than expected
                if (rx_chunk_bitmap.size() < total_chunks) {
                    rx_chunk_bitmap.resize(total_chunks, false);
                }
                
                // Reset bitmap
                // Optimize: Only clear up to total_chunks if we want, but std::fill is fast enough
                std::fill(rx_chunk_bitmap.begin(), rx_chunk_bitmap.end(), false);
            } else if (diff < 0) {
                // Old packet, ignore
                continue;
            }
            // else diff == 0 (Current Frame)

            // Validate Chunk ID
            if (chunk_id >= total_chunks) continue;
            if (chunk_id >= rx_chunk_bitmap.size()) {
                 rx_chunk_bitmap.resize(chunk_id + 1024, false); // Grow aggressively
            }

            // Duplicate Check
            if (rx_chunk_bitmap[chunk_id]) continue;

            // Copy Payload
            size_t offset = chunk_id * LEAP_CHUNK_SIZE;
            if (offset + payload_len <= rx_buffer.size()) {
                std::memcpy(rx_buffer.data() + offset, temp_packet + sizeof(leap_header), payload_len);
            } else {
                // Buffer overflow protection
                continue;
            }

            // Mark Received
            rx_chunk_bitmap[chunk_id] = true;
            rx_chunks_received++;

            // Completion Check
            if (rx_chunks_received == total_chunks) {
                // Frame Complete!
                std::memcpy(data, rx_buffer.data(), size);
                return;
            }
        }
    }
}