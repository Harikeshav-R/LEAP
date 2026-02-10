# Transport Layer (`src/inference/Transport*`)

The transport layer provides a modular networking interface for distributed inference over a **ring topology**. Three implementations are available, all conforming to the abstract `Transport` base class.

## Source Files

| File | Purpose |
|------|---------|
| `Transport.h` | Abstract base class + control message protocol |
| `TcpTransport.h/cpp` | Reliable TCP stream transport |
| `UdpTransport.h/cpp` | Best-effort UDP datagram transport |
| `KernelTransport.h/cpp` | Zero-copy kernel module transport |

---

## Abstract Interface (`Transport.h`)

The base class defines the ring communication API:

```cpp
class Transport {
    virtual void initialize() = 0;

    // Legacy simple send/recv
    virtual void send(const void *data, size_t size) = 0;
    virtual void recv(void *data, size_t size) = 0;

    // Ring topology methods
    virtual void send_next(const void *data, size_t size);  // → next node
    virtual void recv_next(void *data, size_t size);         // ← next node
    virtual void send_prev(const void *data, size_t size);   // → prev node
    virtual void recv_prev(void *data, size_t size);         // ← prev node

    // Zero-copy optimization
    virtual void send_multipart_next(header, header_size, data, data_size);

    // Control channel
    virtual void send_control(const ControlMessage &msg);
    virtual bool recv_control_nonblocking(ControlMessage &msg);
};
```

### Control Protocol

Layer resizing commands are sent as **control messages** embedded in the data stream. They are identified by a magic marker (`0xC0DE`) in the first 2 bytes:

```cpp
struct ControlPacketHeader {
    uint16_t magic;        // 0xC0DE
    ControlMessage msg;
} __attribute__((packed));
```

Control messages are **padded to full packet size** (`sizeof(PacketHeader) + dim × sizeof(float)`) so they flow through `recv_prev()` without special handling. Workers detect them by inspecting the first bytes of each received packet.

**Message Types:**

| Type | Value | Description |
|------|-------|-------------|
| `RESIZE_LAYERS` | 1 | Single-worker resize (split\_layer, end\_layer, is\_tail) |
| `RESIZE_CHAIN` | 2 | Multi-worker resize (array of `LayerRange` for up to 16 workers) |
| `ACK` | 3 | Acknowledgment (propagates back through ring) |

---

## TCP Transport (`TcpTransport`)

A reliable, stream-based transport using standard TCP sockets.

### Connection Setup

1. **Server Side**: Binds to `host:port`, listens for one incoming connection (from previous node in ring).
2. **Client Side**: Connects to `next_host:next_port` (next node in ring).
3. Both connections happen concurrently via `std::thread` to avoid deadlock (both nodes need to simultaneously listen and connect).

### Optimizations

| Optimization | Implementation |
|-------------|---------------|
| **TCP\_NODELAY** | Disables Nagle's algorithm for minimal latency |
| **Large Buffers** | 8 MB `SO_RCVBUF` / `SO_SNDBUF` to absorb bursts |
| **Multipart Send** | `writev()` sends header + data in single syscall (zero-copy scatter/gather) |

### Reliable Send/Recv

Both `send_next()` and `recv_prev()` use retry loops to handle partial reads/writes:
```cpp
while (total < size) {
    ssize_t n = ::write(fd, buf + total, size - total);
    if (n <= 0) throw ...;
    total += n;
}
```

### Non-Blocking Control Receive

`recv_control_nonblocking()` uses `poll()` with a 0ms timeout to check for pending control messages without blocking the data path.

---

## UDP Transport (`UdpTransport`)

A best-effort datagram transport that chunks large tensors into MTU-safe packets.

### Connection Setup

1. Binds a **receive socket** to `host:port`.
2. Creates a **send socket** bound to `host:(port+1)`.
3. Configures large receive buffers (8 MB `SO_RCVBUF`).

### Chunking Protocol

Since activation tensors exceed the typical MTU (~1500 bytes), UDP transport splits them into chunks:

```
LEAP Chunk Header (shared with kernel module):
┌────────────┬──────────┬──────────┬──────────────┬──────────┐
│ magic (4B) │ seq_id   │ chunk_id │ total_chunks │ reserved │
│ "LEAP"     │ (2B)     │ (2B)     │ (2B)         │ (2B)     │
└────────────┴──────────┴──────────┴──────────────┴──────────┘
```

- **Chunk Size**: 1400 bytes payload (fits within 1500-byte Ethernet MTU with header overhead).
- **Sequence ID**: Monotonically increasing per frame, wraps around.
- Each chunk carries `min(LEAP_CHUNK_SIZE, remaining_data)` of payload.

### Reassembly (`recv_prev`)

The receiver reassembles frames using:
1. Track the expected `seq_id`.
2. Maintain a bitmap of received chunks.
3. Copy each chunk to its correct offset: `chunk_id × LEAP_CHUNK_SIZE`.
4. Mark frame complete when all chunks are received.
5. Handle sequence ID wraparound and stale packets via `expected_seq_id` tracking.

### Multipart Send

`send_multipart_next()` merges header + data into a single contiguous buffer, then chunks the combined payload. This avoids an extra copy for each chunk boundary.

### Non-Blocking Control

`send_control()` sends control messages via `sendto()` on the send socket. The `LEAP_MAGIC`-based chunk header distinguishes data from control packets (control packets use `CONTROL_MAGIC` instead).

> **Note:** UDP transport is designed for high-quality LAN environments. There is no retransmission or forward error correction — lost packets result in corrupt frames.

---

## Kernel Transport (`KernelTransport`)

The highest-performance transport, using a custom Linux kernel module (`leap_kmod`) for zero-copy networking.

### How It Works

1. Opens the character device `/dev/leap_tensor`.
2. Memory-maps 16 MB of shared buffer (8 MB RX + 8 MB TX).
3. Sends data by writing to the TX buffer and triggering an `ioctl(LEAP_IOCTL_SEND)`.
4. Receives data by calling `ioctl(LEAP_IOCTL_WAIT_DATA)` which sleeps until a complete frame arrives, then copies from the appropriate RX bank.

### API Mapping

| Operation | Implementation |
|-----------|---------------|
| `send_next()` | `memcpy` to TX buffer → `ioctl(SEND)` |
| `recv_prev()` | `ioctl(WAIT_DATA)` → `memcpy` from RX bank |
| `send_multipart_next()` | Scatter-gather copy to TX → `ioctl(SEND)` |
| `send_control()` | Same as `send_next()` with padded control header |
| `recv_control_nonblocking()` | Not directly supported; handled at higher level |

### Auto-Discovery

The kernel module tracks the source IP/port of each received packet. `KernelTransport::recv_internal()` uses `ioctl(LEAP_IOCTL_GET_BANK_SRC)` to learn the previous node's address for reply routing.

> **See Also:** [Kernel Module](kernel-module.md) for full kernel-side implementation details.

---

## Transport Comparison

| Feature | TCP | UDP | Kernel |
|---------|-----|-----|--------|
| **Reliability** | Guaranteed | Best-effort | Best-effort |
| **Latency** | Medium | Low | Lowest |
| **Zero-Copy** | No (writev) | No | Yes (mmap) |
| **Chunking** | OS-managed | Application | Kernel |
| **Platform** | All | All | Linux only |
| **Non-blocking Control** | Yes (poll) | Via inline check | No |
| **Requires Module** | No | No | Yes (`insmod`) |
