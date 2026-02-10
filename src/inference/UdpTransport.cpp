#include "UdpTransport.h"
#include "../kernel/leap_protocol.h"
#include <sys/socket.h>
#include <arpa/inet.h>
#include <unistd.h>
#include <stdexcept>
#include <iostream>
#include <cstring>
#include <algorithm>
#include <poll.h>
#include <chrono>

// Set to 1 to enable verbose UDP debug logging
#define UDP_DEBUG 1
#if UDP_DEBUG
#define UDP_LOG(fmt, ...) fprintf(stderr, "[UDP] " fmt "\n", ##__VA_ARGS__)
#else
#define UDP_LOG(fmt, ...) ((void)0)
#endif

namespace Inference {
    UdpTransport::UdpTransport(std::string ip, const int port, std::string next_ip, int next_port)
        : ip(std::move(ip)), port(port), next_ip(std::move(next_ip)), next_port(next_port) {
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
        constexpr int buf_size = 8 * 1024 * 1024;
        setsockopt(sockfd, SOL_SOCKET, SO_RCVBUF, &buf_size, sizeof(buf_size));
        setsockopt(sockfd, SOL_SOCKET, SO_SNDBUF, &buf_size, sizeof(buf_size));

        if (bind(sockfd, reinterpret_cast<struct sockaddr *>(&bind_addr), sizeof(bind_addr)) < 0) {
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

    void UdpTransport::drain_stale_packets() {
        // Non-blocking drain of any queued packets from the socket buffer.
        // This prevents stale packets from previous operations from interfering
        // with the next recv_prev call.
        int drained = 0;
        uint8_t discard[2048];
        while (true) {
            struct pollfd pfd{};
            pfd.fd = sockfd;
            pfd.events = POLLIN;
            int ret = poll(&pfd, 1, 0);  // Non-blocking
            if (ret <= 0) break;
            ssize_t len = ::recvfrom(sockfd, discard, sizeof(discard), 0, nullptr, nullptr);
            if (len <= 0) break;
            drained++;
        }
        if (drained > 0) {
            UDP_LOG("Drained %d stale packets from socket buffer", drained);
        }
    }

    void UdpTransport::send(const void *data, const size_t size) {
        send_next(data, size);
    }

    void UdpTransport::recv(void *data, const size_t size) {
        recv_prev(data, size);
    }

    void UdpTransport::send_next(const void *data, const size_t size) {
        if (next_addr.sin_family == 0) throw std::runtime_error("Next address not configured");

        const auto *bytes = static_cast<const uint8_t *>(data);
        if (packet_buffer.size() < sizeof(leap_header)) packet_buffer.resize(sizeof(leap_header));
        auto *hdr = reinterpret_cast<leap_header *>(packet_buffer.data());

        // Split large payloads into LEAP_RX_BANK_SIZE segments so each segment
        // maps to one RX bank in the kernel module. Without this, a single seq_id
        // with >128KB of chunks overflows a bank on kernel transport receivers.
        size_t segment_offset = 0;
        while (segment_offset < size) {
            const size_t segment_size = std::min(size - segment_offset, static_cast<size_t>(LEAP_RX_BANK_SIZE));
            const uint16_t total_chunks = (segment_size + LEAP_CHUNK_SIZE - 1) / LEAP_CHUNK_SIZE;
            uint16_t chunk_idx = 0;
            size_t processed = 0;

            seq_id++;

            UDP_LOG("send_next: seq=%u total_chunks=%u size=%zu to %s:%d",
                    seq_id, total_chunks, segment_size,
                    inet_ntoa(next_addr.sin_addr), ntohs(next_addr.sin_port));

            while (processed < segment_size) {
                size_t chunk_len = std::min(segment_size - processed, static_cast<size_t>(LEAP_CHUNK_SIZE));

                hdr->magic = htonl(LEAP_MAGIC);
                hdr->seq_id = htons(seq_id);
                hdr->chunk_id = htons(chunk_idx);
                hdr->total_chunks = htons(total_chunks);

                struct iovec iov[2];
                iov[0].iov_base = packet_buffer.data();
                iov[0].iov_len = sizeof(leap_header);
                iov[1].iov_base = const_cast<uint8_t *>(bytes + segment_offset + processed);
                iov[1].iov_len = chunk_len;

                struct msghdr msg{};
                msg.msg_name = &next_addr;
                msg.msg_namelen = sizeof(next_addr);
                msg.msg_iov = iov;
                msg.msg_iovlen = 2;

                if (sendmsg(sockfd, &msg, 0) < 0) throw std::runtime_error("UDP send next failed");

                processed += chunk_len;
                chunk_idx++;

                // Pacing: brief pause between chunks to prevent receiver socket buffer
                // overflow. Virtual NICs (Mac host to Linux VM) are especially prone
                // to dropping back-to-back UDP packets.
                if (total_chunks > 1) {
                    if (total_chunks > 32 && chunk_idx % 32 == 0) {
                        usleep(200);   // Longer pause for large transfers
                    } else {
                        usleep(50);    // 50μs between chunks — negligible latency, prevents drops
                    }
                }
            }
            segment_offset += segment_size;
        }
    }

    void UdpTransport::send_multipart_next(const void *header, const size_t header_size, const void *data,
                                           const size_t data_size) {
        if (next_addr.sin_family == 0) throw std::runtime_error("Next address not configured");

        // Merge header + data and delegate to send_next, which handles
        // bank-aligned splitting for kernel transport compatibility.
        const size_t total_size = header_size + data_size;
        std::vector<uint8_t> buffer(total_size);
        std::memcpy(buffer.data(), header, header_size);
        std::memcpy(buffer.data() + header_size, data, data_size);
        send_next(buffer.data(), total_size);
    }

    void UdpTransport::recv_next(void *data, size_t size) {
        throw std::runtime_error("recv_next not supported in Ring");
    }

    void UdpTransport::send_prev(const void *data, size_t size) {
        throw std::runtime_error("send_prev not supported in Ring");
    }

    void UdpTransport::recv_prev(void *data, const size_t size) {
        auto *out_bytes = static_cast<uint8_t *>(data);

        // Overall deadline: 10s from start of recv_prev.
        // Per-poll timeout: 500ms — short enough to detect partial delivery
        // instead of blocking on a single 10s poll where one lost packet = total hang.
        constexpr int OVERALL_TIMEOUT_MS = 10000;
        constexpr int POLL_TIMEOUT_MS = 500;

        UDP_LOG("recv_prev: waiting for %zu bytes (%zu chunks)",
                size, (size + LEAP_CHUNK_SIZE - 1) / LEAP_CHUNK_SIZE);

        auto start_time = std::chrono::steady_clock::now();

        // Receive in LEAP_RX_BANK_SIZE segments to match the bank-aligned
        // splitting in send_next. Each segment has its own seq_id.
        size_t total_received = 0;
        while (total_received < size) {
            const size_t segment_size = std::min(size - total_received, static_cast<size_t>(LEAP_RX_BANK_SIZE));
            const size_t expected_chunks = (segment_size + LEAP_CHUNK_SIZE - 1) / LEAP_CHUNK_SIZE;
            std::vector<bool> received_chunks(expected_chunks, false);
            size_t chunks_count = 0;
            int32_t active_seq_id = -1;
            int stale_packets = 0;

            while (chunks_count < expected_chunks) {
                // Check overall deadline
                auto now = std::chrono::steady_clock::now();
                auto elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(now - start_time).count();
                if (elapsed_ms >= OVERALL_TIMEOUT_MS) {
                    UDP_LOG("recv_prev TIMEOUT: %zu/%zu chunks, active_seq=%d, stale_dropped=%d, elapsed=%lldms",
                            chunks_count, expected_chunks, active_seq_id, stale_packets, elapsed_ms);
                    throw std::runtime_error("UDP recv_prev timeout: received " + 
                        std::to_string(chunks_count) + "/" + std::to_string(expected_chunks) + 
                        " chunks (" + std::to_string(size) + " bytes expected)");
                }

                sockaddr_in src_addr{};
                socklen_t addr_len = sizeof(src_addr);

                // Reuse packet_buffer for receive
                if (packet_buffer.size() < 2048) packet_buffer.resize(2048);

                // Short poll: allows us to check the overall deadline frequently
                // instead of blocking for the full 10s on a single poll call.
                struct pollfd pfd{};
                pfd.fd = sockfd;
                pfd.events = POLLIN;
                int poll_ret = poll(&pfd, 1, POLL_TIMEOUT_MS);
                if (poll_ret == 0) {
                    // Short timeout: no packet in 500ms. Log and retry until overall deadline.
                    if (chunks_count > 0) {
                        UDP_LOG("recv_prev: partial receive (%zu/%zu chunks), still waiting...",
                                chunks_count, expected_chunks);
                    }
                    continue;  // Check overall deadline at top of loop
                }
                if (poll_ret < 0) throw std::runtime_error("UDP poll failed");

                const ssize_t len = recvfrom(sockfd, packet_buffer.data(), packet_buffer.size(), 0,
                                             reinterpret_cast<struct sockaddr *>(&src_addr), &addr_len);

                if (len < 0) throw std::runtime_error("UDP recv prev failed");

                // Learn Prev Addr (optional, just logging or verification)
                if (!prev_addr_set) {
                    prev_addr = src_addr;
                    prev_addr_set = true;
                }

                if (static_cast<size_t>(len) < sizeof(leap_header)) {
                    UDP_LOG("recv_prev: packet too small (%zd bytes) from %s:%d",
                            len, inet_ntoa(src_addr.sin_addr), ntohs(src_addr.sin_port));
                    continue;
                }

                auto *hdr = reinterpret_cast<leap_header *>(packet_buffer.data());
                if (ntohl(hdr->magic) != LEAP_MAGIC) {
                    UDP_LOG("recv_prev: non-LEAP packet (magic=0x%08x, len=%zd) from %s:%d",
                            ntohl(hdr->magic), len,
                            inet_ntoa(src_addr.sin_addr), ntohs(src_addr.sin_port));
                    continue;
                }

                const uint16_t seq = ntohs(hdr->seq_id);
                const uint16_t chunk = ntohs(hdr->chunk_id);
                const uint16_t total = ntohs(hdr->total_chunks);

                if (active_seq_id == -1) {
                    active_seq_id = seq;
                    UDP_LOG("recv_prev: locked onto seq=%u (total_chunks=%u) from %s:%d",
                            seq, total, inet_ntoa(src_addr.sin_addr), ntohs(src_addr.sin_port));
                } else if (seq != static_cast<uint16_t>(active_seq_id)) {
                    // Check if this is a NEWER seq_id (sender moved on).
                    // If so, reset and start collecting the new sequence.
                    auto seq_diff = static_cast<int16_t>(seq - static_cast<uint16_t>(active_seq_id));
                    if (seq_diff > 0) {
                        // Newer sequence - reset and start fresh
                        UDP_LOG("recv_prev: newer seq=%u arrived (was %d), resetting (%zu chunks lost)",
                                seq, active_seq_id, chunks_count);
                        active_seq_id = seq;
                        std::fill(received_chunks.begin(), received_chunks.end(), false);
                        chunks_count = 0;
                    } else {
                        // Older/stale sequence - discard
                        stale_packets++;
                        UDP_LOG("recv_prev: stale seq=%u (active=%d), dropping (stale_count=%d)",
                                seq, active_seq_id, stale_packets);
                        continue;
                    }
                }

                if (chunk >= expected_chunks) continue;
                if (received_chunks[chunk]) continue;

                size_t payload_len = len - sizeof(leap_header);
                size_t offset = total_received + chunk * LEAP_CHUNK_SIZE;

                if (offset + payload_len > size) payload_len = size - offset;

                std::memcpy(out_bytes + offset, packet_buffer.data() + sizeof(leap_header), payload_len);
                received_chunks[chunk] = true;
                chunks_count++;
            }
            UDP_LOG("recv_prev: segment complete (seq=%d, %zu chunks, %d stale dropped)",
                    active_seq_id, chunks_count, stale_packets);
            total_received += segment_size;
        }
    }

    void UdpTransport::send_control(const ControlMessage &msg) {
        if (next_addr.sin_family == 0) throw std::runtime_error("Next address not configured for control");

        // For UDP, use the same chunked protocol as data packets to ensure ordering.
        // Control messages are padded to packet_size_ and sent via send_next, which
        // uses the leap_header chunking. Workers will detect them inline via CONTROL_MAGIC.
        ControlPacketHeader pkt{CONTROL_MAGIC, msg};
        
        UDP_LOG("send_control: type=%d, packet_size=%zu",
                static_cast<int>(msg.type), packet_size_);
        
        if (packet_size_ > 0 && packet_size_ >= sizeof(pkt)) {
            // Pad to full packet size and send through chunking protocol
            std::vector<char> buffer(packet_size_, 0);
            std::memcpy(buffer.data(), &pkt, sizeof(pkt));
            send_next(buffer.data(), buffer.size());
        } else {
            // Fallback: send raw (may not work reliably)
            send_next(&pkt, sizeof(pkt));
        }
    }

    bool UdpTransport::recv_control_nonblocking(ControlMessage &msg) {
        // Control messages are now sent via the chunked protocol (send_next) for ordering.
        // They will be received through recv_prev and detected inline in worker_loop
        // by checking for CONTROL_MAGIC at the start of the received packet.
        // Non-blocking control receive is not needed/supported with this approach.
        (void)msg;
        return false;
    }
}