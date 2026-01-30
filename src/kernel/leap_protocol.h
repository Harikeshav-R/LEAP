#ifndef LEAP_PROTOCOL_H
#define LEAP_PROTOCOL_H

#ifdef __KERNEL__
#include <linux/types.h>
#else
#include <stdint.h>
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

// Buffer Size (64KB - enough for 4096 dim float tensor)
#define LEAP_BUFFER_SIZE (64 * 1024)
#define LEAP_CHUNK_SIZE 1400

// Packet Header structure
struct leap_header {
    __be32 magic;
    __be16 seq_id;
    __u8 chunk_id;
    __u8 total_chunks;
} __attribute__((packed));

#endif // LEAP_PROTOCOL_H
