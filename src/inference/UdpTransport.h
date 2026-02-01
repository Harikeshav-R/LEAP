#ifndef LEAP_UDP_TRANSPORT_H
#define LEAP_UDP_TRANSPORT_H

#include "Transport.h"
#include <string>
#include <vector>
#include <netinet/in.h>

namespace Inference {
    class UdpTransport : public Transport {
    public:
        UdpTransport(std::string ip, int port, bool is_server);

        ~UdpTransport() override;

        void initialize() override;

        void send(const void *data, size_t size) override;

        void recv(void *data, size_t size) override;

    private:
        std::string ip;
        int port;
        bool is_server;
        int sockfd = -1;
        sockaddr_in dest_addr{};
        uint16_t tx_seq_id = 0; // Renamed for clarity

        // RX State Tracking
        uint16_t rx_active_seq_id = 0;
        std::vector<bool> rx_chunk_bitmap;
        size_t rx_chunks_received = 0;
        std::vector<uint8_t> rx_buffer; // Internal buffer to assemble the frame
    };
}

#endif //LEAP_UDP_TRANSPORT_H