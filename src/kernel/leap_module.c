#include <linux/module.h>
#include <linux/kernel.h>
#include <linux/init.h>
#include <linux/fs.h>
#include <linux/cdev.h>
#include <linux/uaccess.h>
#include <linux/slab.h>
#include <linux/mm.h>
#include <linux/netfilter.h>
#include <linux/netfilter_ipv4.h>
#include <linux/ip.h>
#include <linux/udp.h>
#include <linux/wait.h>
#include <linux/sched.h>
#include <linux/net.h>
#include <linux/in.h>
#include <linux/vmalloc.h>
#include "leap_protocol.h"

MODULE_LICENSE("GPL");
MODULE_AUTHOR("Harikeshav R");
MODULE_DESCRIPTION("Low-Latency Tensor Transport for LEAP Inference");
MODULE_VERSION("0.4");

// Global State
static int major_number;
static struct class* leap_class = NULL;
static struct device* leap_device = NULL;
static struct cdev leap_cdev;

// Memory Buffer (16MB Total: Lower 8MB RX, Upper 8MB TX)
static void *leap_buffer = NULL;
static unsigned long leap_buffer_size = LEAP_BUFFER_SIZE;
static unsigned long total_alloc_size; // 2 * LEAP_BUFFER_SIZE

// Synchronization
static DECLARE_WAIT_QUEUE_HEAD (leap_wait_queue);
static atomic_t data_ready = ATOMIC_INIT(0);
static atomic_t open_count = ATOMIC_INIT(0);

// RX State
static spinlock_t rx_lock;
static uint16_t active_seq_id = 0;
static atomic_t chunks_received = ATOMIC_INIT(0);

// Networking (TX)
static struct socket *tx_socket = NULL;
static struct sockaddr_in dest_addr;
static unsigned int dest_ip = 0;
static uint16_t dest_port = htons(LEAP_PORT);
static uint16_t listening_port = htons(LEAP_PORT);
static atomic_t global_seq_id = ATOMIC_INIT(0);

// Netfilter Hook (RX)
static struct nf_hook_ops leap_nf_ops;

static int leap_dev_open(struct inode *, struct file *);
static int leap_dev_release(struct inode *, struct file *);
static ssize_t leap_dev_write(struct file *, const char __user *, size_t, loff_t *);
static long leap_dev_ioctl(struct file *, unsigned int, unsigned long);
static int leap_dev_mmap(struct file *, struct vm_area_struct *);
static unsigned int leap_nf_hook(void *priv, struct sk_buff *skb, const struct nf_hook_state *state);

static struct file_operations fops = {
    .owner = THIS_MODULE,
    .open = leap_dev_open,
    .release = leap_dev_release,
    .write = leap_dev_write,
    .unlocked_ioctl = leap_dev_ioctl,
    .mmap = leap_dev_mmap,
};

static int send_udp_chunk(void *data, size_t len, uint16_t seq, uint16_t chunk, uint16_t total) {
    struct msghdr msg;
    struct kvec vec[2];
    struct leap_header hdr;
    int ret;

    if (unlikely(!tx_socket || dest_ip == 0)) return -ENOTCONN;

    memset(&msg, 0, sizeof(msg));
    dest_addr.sin_port = dest_port;
    msg.msg_name = &dest_addr;
    msg.msg_namelen = sizeof(dest_addr);

    hdr.magic = cpu_to_be32(LEAP_MAGIC);
    hdr.seq_id = cpu_to_be16(seq);
    hdr.chunk_id = cpu_to_be16(chunk);
    hdr.total_chunks = cpu_to_be16(total);

    vec[0].iov_base = &hdr;
    vec[0].iov_len = sizeof(hdr);
    vec[1].iov_base = data;
    vec[1].iov_len = len;

    ret = kernel_sendmsg(tx_socket, &msg, vec, 2, sizeof(hdr) + len);
    if (unlikely(ret < 0)) {
        if (printk_ratelimit())
            printk(KERN_ERR "LEAP: TX Failed! Error: %d\n", ret);
    }
    return ret;
}

static unsigned int leap_nf_hook(void *priv, struct sk_buff *skb, const struct nf_hook_state *state) {
    struct iphdr *iph;
    struct udphdr *udph;
    struct leap_header *lhdr;
    int payload_offset, payload_len, leap_offset;

    if (unlikely(!skb)) return NF_ACCEPT;
    if (unlikely(atomic_read(&open_count) == 0)) return NF_ACCEPT;
    if (unlikely(!pskb_may_pull(skb, sizeof(struct iphdr)))) return NF_ACCEPT;

    iph = ip_hdr(skb);
    if (!iph || iph->protocol != IPPROTO_UDP) return NF_ACCEPT;

    int ip_hlen = iph->ihl * 4;
    if (unlikely(!pskb_may_pull(skb, ip_hlen + sizeof(struct udphdr)))) return NF_ACCEPT;

    iph = ip_hdr(skb);
    udph = (struct udphdr *) ((unsigned char *) iph + ip_hlen);
    if (udph->dest != listening_port) return NF_ACCEPT;

    if (unlikely(ntohs(udph->len) < sizeof(struct udphdr) + sizeof(struct leap_header))) return NF_ACCEPT;

    payload_offset = ip_hlen + sizeof(struct udphdr);
    payload_len = ntohs(udph->len) - sizeof(struct udphdr);

    if (unlikely(!pskb_may_pull(skb, payload_offset + sizeof(struct leap_header)))) return NF_ACCEPT;

    iph = ip_hdr(skb);
    lhdr = (struct leap_header *) ((unsigned char *) iph + payload_offset);

    if (unlikely(lhdr->magic != cpu_to_be32(LEAP_MAGIC))) return NF_ACCEPT;

    uint16_t chunk_id = ntohs(lhdr->chunk_id);
    uint16_t total_chunks = ntohs(lhdr->total_chunks);
    uint16_t seq_id = ntohs(lhdr->seq_id);

    if (unlikely(total_chunks == 0 || chunk_id >= total_chunks)) return NF_ACCEPT;

    if (dest_port != udph->source) dest_port = udph->source;

    leap_offset = chunk_id * LEAP_CHUNK_SIZE;
    int data_len = payload_len - sizeof(struct leap_header);
    if (unlikely(data_len < 0)) return NF_ACCEPT;

    spin_lock(&rx_lock);
    int16_t diff = (int16_t)(seq_id - active_seq_id);
    if (diff > 0) {
        active_seq_id = seq_id;
        atomic_set(&chunks_received, 0);
    } else if (diff < 0) {
        spin_unlock(&rx_lock);
        return NF_ACCEPT;
    }
    spin_unlock(&rx_lock);

    if (likely(leap_offset + data_len <= leap_buffer_size)) {
        int abs_offset = payload_offset + sizeof(struct leap_header);
        
        // Optimization: Prefetch destination cache line for writing
        prefetchw(leap_buffer + leap_offset);

        if (skb_copy_bits(skb, abs_offset, leap_buffer + leap_offset, data_len) < 0) return NF_ACCEPT;

        if (atomic_inc_return(&chunks_received) == total_chunks) {
            atomic_set(&data_ready, 1);
            wake_up_interruptible(&leap_wait_queue);
        }
    }

    consume_skb(skb);
    return NF_STOLEN;
}

static int leap_dev_open(struct inode *inodep, struct file *filep) {
    if (atomic_read(&open_count) > 0) return -EBUSY;
    atomic_inc(&open_count);
    pr_alert("LEAP: Device opened by PID %d\n", current->pid);
    return 0;
}

static int leap_dev_release(struct inode *inodep, struct file *filep) {
    atomic_dec(&open_count);
    listening_port = htons(LEAP_PORT);
    return 0;
}

static ssize_t leap_dev_write(struct file *filep, const char __user *buffer, size_t len, loff_t *offset) {
    size_t processed = 0;
    uint16_t total_chunks, chunk_idx = 0;
    void *kbuf;

    if (len > leap_buffer_size) return -EMSGSIZE;
    if (!tx_socket) return -EIO;

    total_chunks = (len + LEAP_CHUNK_SIZE - 1) / LEAP_CHUNK_SIZE;
    kbuf = vmalloc(len);
    if (!kbuf) return -ENOMEM;

    if (copy_from_user(kbuf, buffer, len)) {
        vfree(kbuf);
        return -EFAULT;
    }

    uint16_t current_seq = (uint16_t)atomic_inc_return(&global_seq_id);
    while (processed < len) {
        size_t chunk_len = (processed + LEAP_CHUNK_SIZE > len) ? (len - processed) : LEAP_CHUNK_SIZE;
        send_udp_chunk(kbuf + processed, chunk_len, current_seq, chunk_idx++, total_chunks);
        processed += chunk_len;
        cond_resched();
    }
    vfree(kbuf);
    return len;
}

static long leap_dev_ioctl(struct file *filep, unsigned int cmd, unsigned long arg) {
    if (cmd == LEAP_IOCTL_WAIT_DATA) {
        int i;
        for (i = 0; i < 5000; i++) {
            if (atomic_read(&data_ready) != 0) {
                atomic_set(&data_ready, 0);
                return 0;
            }
            cpu_relax();
        }
        if (wait_event_interruptible(leap_wait_queue, atomic_read(&data_ready) != 0)) return -ERESTARTSYS;
        atomic_set(&data_ready, 0);
        return 0;
    } else if (cmd == LEAP_IOCTL_SET_DEST) {
        if (copy_from_user(&dest_ip, (unsigned int __user *)arg, sizeof(dest_ip))) return -EFAULT;
        memset(&dest_addr, 0, sizeof(dest_addr));
        dest_addr.sin_family = AF_INET;
        dest_addr.sin_port = htons(LEAP_PORT);
        dest_addr.sin_addr.s_addr = dest_ip;
        return 0;
    } else if (cmd == LEAP_IOCTL_SET_PORT) {
        unsigned short port_arg;
        if (copy_from_user(&port_arg, (unsigned short __user *)arg, sizeof(port_arg))) return -EFAULT;
        listening_port = htons(port_arg);
        if (dest_port == htons(LEAP_PORT)) dest_port = htons(port_arg);
        return 0;
    } else if (cmd == LEAP_IOCTL_SEND) {
        unsigned int data_len;
        if (copy_from_user(&data_len, (unsigned int __user *)arg, sizeof(data_len))) return -EFAULT;
        if (data_len > leap_buffer_size || !tx_socket) return -EINVAL;

        uint16_t total_chunks = (data_len + LEAP_CHUNK_SIZE - 1) / LEAP_CHUNK_SIZE;
        uint16_t current_seq = (uint16_t)atomic_inc_return(&global_seq_id);
        size_t processed = 0;
        uint16_t chunk_idx = 0;
        void *tx_buffer_start = leap_buffer + leap_buffer_size;

        while (processed < data_len) {
            size_t chunk_len = (processed + LEAP_CHUNK_SIZE > data_len) ? (data_len - processed) : LEAP_CHUNK_SIZE;
            send_udp_chunk(tx_buffer_start + processed, chunk_len, current_seq, chunk_idx++, total_chunks);
            processed += chunk_len;
            cond_resched();
        }
        return data_len;
    }
    return -EINVAL;
}

static int leap_dev_mmap(struct file *filp, struct vm_area_struct *vma) {
    unsigned long size = vma->vm_end - vma->vm_start;
    int ret;

    pr_alert("LEAP: mmap called. Size: %lu, TotalAlloc: %lu\n", size, total_alloc_size);

    if (size > total_alloc_size) {
        pr_alert("LEAP: mmap failed! Request too large.\n");
        return -EINVAL;
    }
    
    ret = remap_vmalloc_range(vma, leap_buffer, 0);
    if (ret) {
        pr_alert("LEAP: remap_vmalloc_range failed with error %d\n", ret);
    }
    return ret;
}

static int __init leap_init(void) {
    int ret;
    dev_t dev_no;
    
    pr_alert("LEAP: Initializing module v0.5 (Double Buffered 16MB)\n");

    total_alloc_size = leap_buffer_size * 2;
    leap_buffer = vmalloc(total_alloc_size);
    if (!leap_buffer) return -ENOMEM;
    memset(leap_buffer, 0, total_alloc_size);
    spin_lock_init(&rx_lock);

    ret = alloc_chrdev_region(&dev_no, 0, 1, LEAP_DEVICE_NAME);
    if (ret < 0) goto err_buffer;
    major_number = MAJOR(dev_no);
    leap_class = class_create(LEAP_CLASS_NAME);
    if (IS_ERR(leap_class)) { ret = PTR_ERR(leap_class); goto err_chrdev; }
    leap_device = device_create(leap_class, NULL, dev_no, NULL, LEAP_DEVICE_NAME);
    if (IS_ERR(leap_device)) { ret = PTR_ERR(leap_device); goto err_class; }
    cdev_init(&leap_cdev, &fops);
    if (cdev_add(&leap_cdev, dev_no, 1) < 0) goto err_device;

    leap_nf_ops.hook = leap_nf_hook;
    leap_nf_ops.pf = PF_INET;
    leap_nf_ops.hooknum = NF_INET_PRE_ROUTING;
    leap_nf_ops.priority = NF_IP_PRI_FIRST;
    if (nf_register_net_hook(&init_net, &leap_nf_ops) < 0) goto err_cdev;

    if (sock_create_kern(&init_net, PF_INET, SOCK_DGRAM, IPPROTO_UDP, &tx_socket) < 0) goto err_nf;
    
    // Optimization: Disable UDP Checksums for performance
    if (tx_socket->sk) {
        tx_socket->sk->sk_no_check_tx = 1;
    }

    return 0;

err_nf: nf_unregister_net_hook(&init_net, &leap_nf_ops);
err_cdev: cdev_del(&leap_cdev);
err_device: device_destroy(leap_class, dev_no);
err_class: class_destroy(leap_class);
err_chrdev: unregister_chrdev_region(dev_no, 1);
err_buffer: vfree(leap_buffer);
    return ret;
}

static void __exit leap_exit(void) {
    dev_t dev_no = MKDEV(major_number, 0);
    if (tx_socket) sock_release(tx_socket);
    nf_unregister_net_hook(&init_net, &leap_nf_ops);
    cdev_del(&leap_cdev);
    device_destroy(leap_class, dev_no);
    class_destroy(leap_class);
    unregister_chrdev_region(dev_no, 1);
    vfree(leap_buffer);
}

module_init(leap_init);
module_exit(leap_exit);