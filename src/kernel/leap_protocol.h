#ifndef LEAP_PROTOCOL_H
#define LEAP_PROTOCOL_H

#ifdef __KERNEL__
#include <linux/types.h>
#else
#include <stdint.h>
#include <sys/ioctl.h>
// Userspace compatibility definitions
typedef uint32_t __be32;
typedef uint16_t __be16;
typedef uint8_t __u8;
#endif

#define LEAP_MAGIC 0x4C454150 // "LEAP"
#define LEAP_PORT 9999
#define LEAP_DEVICE_NAME "leap_tensor"
#define LEAP_CLASS_NAME "leap"

// IOCTL Definitions
#define LEAP_IOCTL_MAGIC 'k'
#define LEAP_IOCTL_WAIT_DATA _IO(LEAP_IOCTL_MAGIC, 1)
#define LEAP_IOCTL_SET_DEST  _IOW(LEAP_IOCTL_MAGIC, 2, unsigned int) // Set Dest IP (u32)
#define LEAP_IOCTL_SET_PORT  _IOW(LEAP_IOCTL_MAGIC, 3, unsigned short) // Set Listen Port (u16)
#define LEAP_IOCTL_SET_TX_PORT _IOW(LEAP_IOCTL_MAGIC, 5, unsigned short) // Set TX Port (u16)

// Buffer Size (8MB)
#define LEAP_BUFFER_SIZE (8 * 1024 * 1024)
#define LEAP_RX_BANK_SIZE (LEAP_BUFFER_SIZE / 2)
#define LEAP_CHUNK_SIZE 1400

// Packet Header structure
struct leap_header {
    __be32 magic;
    __be16 seq_id;
    __be16 chunk_id;
    __be16 total_chunks;
} __attribute__((packed));

/*
 * Note: Standard write() is deprecated due to extra memory copy.
 * Use mmap() + LEAP_IOCTL_SEND for zero-copy transmission.
 */
#define LEAP_IOCTL_SEND _IOW(LEAP_IOCTL_MAGIC, 4, unsigned int) // Trigger Send (arg = size)

#endif // LEAP_PROTOCOL_H