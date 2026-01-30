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
#include "leap_protocol.h"

MODULE_LICENSE("GPL");
MODULE_AUTHOR("LEAP Agent");
MODULE_DESCRIPTION("Low-Latency Tensor Transport for LEAP Inference");
MODULE_VERSION("0.2");

// Global State
static int major_number;
static struct class* leap_class = NULL;
static struct device* leap_device = NULL;
static struct cdev leap_cdev;

// Memory Buffer (Ring Buffer for RX)
static void* leap_buffer = NULL;
static unsigned long leap_buffer_size = LEAP_BUFFER_SIZE;

// Synchronization
static DECLARE_WAIT_QUEUE_HEAD(leap_wait_queue);
static atomic_t data_ready = ATOMIC_INIT(0);

// Networking (TX)
static struct socket *tx_socket = NULL;
static struct sockaddr_in dest_addr;
static unsigned int dest_ip = 0; // Stored in Network Byte Order
static uint16_t dest_port = LEAP_PORT; // Default, but updated by RX
static uint16_t global_seq_id = 0;

// Netfilter Hook (RX)
static struct nf_hook_ops leap_nf_ops;

// ... (prototypes) ...
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

// --- Helper: Send UDP Chunk ---
static int send_udp_chunk(void *data, size_t len, uint16_t seq, uint8_t chunk, uint8_t total) {
    struct msghdr msg;
    struct kvec vec[2]; // Header + Data
    struct leap_header hdr;
    int ret;

    if (!tx_socket || dest_ip == 0) return -ENOTCONN;

    memset(&msg, 0, sizeof(msg));
    
    // Update port dynamically
    dest_addr.sin_port = dest_port; 
    
    msg.msg_name = &dest_addr;
    msg.msg_namelen = sizeof(dest_addr);

    // Prepare Header
    hdr.magic = cpu_to_be32(LEAP_MAGIC);
    hdr.seq_id = cpu_to_be16(seq);
    hdr.chunk_id = chunk;
    hdr.total_chunks = total;

    vec[0].iov_base = &hdr;
    vec[0].iov_len = sizeof(hdr);
    vec[1].iov_base = data;
    vec[1].iov_len = len;

    // Kernel Send
    ret = kernel_sendmsg(tx_socket, &msg, vec, 2, sizeof(hdr) + len);
    return ret;
}

// --- Netfilter Hook Logic (RX) ---
static unsigned int leap_nf_hook(void *priv, struct sk_buff *skb, const struct nf_hook_state *state) {
    struct iphdr *iph;
    struct udphdr *udph;
    struct leap_header *lhdr;
    unsigned char *payload;
    int payload_len;
    int offset;

    if (!skb) return NF_ACCEPT;

    iph = ip_hdr(skb);
    if (!iph) return NF_ACCEPT;

    if (iph->protocol != IPPROTO_UDP) return NF_ACCEPT;

    // In PRE_ROUTING, transport_header might not be set. Calculate manually.
    udph = (struct udphdr *)((unsigned char *)iph + (iph->ihl * 4));

    // Basic bounds check
    if (skb->len < (iph->ihl * 4) + sizeof(struct udphdr)) return NF_ACCEPT;

    if (ntohs(udph->dest) != LEAP_PORT) return NF_ACCEPT;

    // Calculate payload position
    payload = (unsigned char *)udph + sizeof(struct udphdr);
    payload_len = ntohs(udph->len) - sizeof(struct udphdr);

    if (payload_len < sizeof(struct leap_header)) return NF_ACCEPT;

    lhdr = (struct leap_header *)payload;
    if (lhdr->magic != cpu_to_be32(LEAP_MAGIC)) return NF_ACCEPT;

    // --- LEAP PACKET DETECTED ---
    
    // Capture Source Port for Reply
    if (dest_port != udph->source) {
        dest_port = udph->source;
        // printk(KERN_INFO "LEAP: Learned destination port: %d\n", ntohs(dest_port));
    }
    
    offset = lhdr->chunk_id * LEAP_CHUNK_SIZE;

    payload_len -= sizeof(struct leap_header);
    
    // Safety check bounds
    if (offset + payload_len <= leap_buffer_size) {
        // Copy payload to kernel buffer
        // Note: In IRQ context, memcpy is safe for kernel memory
        memcpy(leap_buffer + offset, payload + sizeof(struct leap_header), payload_len);
        
        // If last chunk, wake up reader
        if (lhdr->chunk_id == lhdr->total_chunks - 1) {
             atomic_set(&data_ready, 1);
             wake_up_interruptible(&leap_wait_queue);
        }
    }

    // Stolen means we consumed the packet, OS stack won't see it.
    return NF_STOLEN; 
}

// --- File Operations ---

static int leap_dev_open(struct inode *inodep, struct file *filep) {
    return 0;
}

static int leap_dev_release(struct inode *inodep, struct file *filep) {
    return 0;
}

// Write: Userspace writes a full tensor buffer here. We fragment and send.
static ssize_t leap_dev_write(struct file *filep, const char __user *buffer, size_t len, loff_t *offset) {
    size_t processed = 0;
    uint8_t total_chunks = (len + LEAP_CHUNK_SIZE - 1) / LEAP_CHUNK_SIZE;
    uint8_t chunk_idx = 0;
    void *kbuf;
    
    if (!tx_socket) return -EIO;

    // Allocate temp buffer for the write operation
    kbuf = kmalloc(len, GFP_KERNEL);
    if (!kbuf) return -ENOMEM;

    if (copy_from_user(kbuf, buffer, len)) {
        kfree(kbuf);
        return -EFAULT;
    }

    global_seq_id++;

    while (processed < len) {
        size_t chunk_len = LEAP_CHUNK_SIZE;
        if (processed + chunk_len > len) {
            chunk_len = len - processed;
        }

        send_udp_chunk(kbuf + processed, chunk_len, global_seq_id, chunk_idx, total_chunks);

        processed += chunk_len;
        chunk_idx++;
        // Small yield to prevent network card saturation if needed?
        // cond_resched(); 
    }

    kfree(kbuf);
    return len;
}

static long leap_dev_ioctl(struct file *filep, unsigned int cmd, unsigned long arg) {
    if (cmd == LEAP_IOCTL_WAIT_DATA) {
        // Blocking wait until data_ready is non-zero
        if (wait_event_interruptible(leap_wait_queue, atomic_read(&data_ready) != 0)) {
            return -ERESTARTSYS; // Signal interrupted
        }
        atomic_set(&data_ready, 0); // Reset
        return 0;
    } else if (cmd == LEAP_IOCTL_SET_DEST) {
        // Arg is IPv4 address (u32)
        if (copy_from_user(&dest_ip, (unsigned int __user *)arg, sizeof(dest_ip))) {
            return -EFAULT;
        }
        memset(&dest_addr, 0, sizeof(dest_addr));
        dest_addr.sin_family = AF_INET;
        dest_addr.sin_port = htons(LEAP_PORT);
        dest_addr.sin_addr.s_addr = dest_ip;
        printk(KERN_INFO "LEAP: Destination set to %pI4\n", &dest_ip);
        return 0;
    }
    return -EINVAL;
}

static int leap_dev_mmap(struct file *filp, struct vm_area_struct *vma) {
    unsigned long pfn;
    unsigned long size = vma->vm_end - vma->vm_start;

    if (size > leap_buffer_size) return -EINVAL;

    // Get the physical page frame number of our kernel buffer
    pfn = virt_to_phys(leap_buffer) >> PAGE_SHIFT;

    // Remap user space range to point to this PFN
    if (remap_pfn_range(vma, vma->vm_start, pfn, size, vma->vm_page_prot)) {
        return -EAGAIN;
    }

    return 0;
}

// --- Module Init/Exit ---

static int __init leap_init(void) {
    int ret;
    dev_t dev_no;

    printk(KERN_INFO "LEAP: Initializing module\n");

    // 1. Allocate Kernel Buffer
    leap_buffer = kmalloc(leap_buffer_size, GFP_KERNEL | __GFP_COMP); 
    if (!leap_buffer) {
        return -ENOMEM;
    }
    // Mark pages reserved
    {
        int i;
        for (i = 0; i < leap_buffer_size; i += PAGE_SIZE) {
            SetPageReserved(virt_to_page(((unsigned long)leap_buffer) + i));
        }
    }
    memset(leap_buffer, 0, leap_buffer_size);

    // 2. Register Char Device
    ret = alloc_chrdev_region(&dev_no, 0, 1, LEAP_DEVICE_NAME);
    if (ret < 0) goto err_buffer;
    major_number = MAJOR(dev_no);

    leap_class = class_create(LEAP_CLASS_NAME);
    if (IS_ERR(leap_class)) {
        ret = PTR_ERR(leap_class);
        goto err_chrdev;
    }

    leap_device = device_create(leap_class, NULL, dev_no, NULL, LEAP_DEVICE_NAME);
    if (IS_ERR(leap_device)) {
        ret = PTR_ERR(leap_device);
        goto err_class;
    }

    cdev_init(&leap_cdev, &fops);
    ret = cdev_add(&leap_cdev, dev_no, 1);
    if (ret < 0) goto err_device;

    // 3. Register Netfilter Hook
    leap_nf_ops.hook = leap_nf_hook;
    leap_nf_ops.pf = PF_INET; // IPv4
    leap_nf_ops.hooknum = NF_INET_PRE_ROUTING;
    leap_nf_ops.priority = NF_IP_PRI_FIRST; // High priority
    
    ret = nf_register_net_hook(&init_net, &leap_nf_ops);
    if (ret < 0) goto err_cdev;

    // 4. Create TX Socket
    ret = sock_create_kern(&init_net, PF_INET, SOCK_DGRAM, IPPROTO_UDP, &tx_socket);
    if (ret < 0) {
        printk(KERN_ERR "LEAP: Failed to create TX socket\n");
        goto err_nf;
    }

    printk(KERN_INFO "LEAP: Module loaded. /dev/%s created with major %d\n", LEAP_DEVICE_NAME, major_number);
    return 0;

err_nf:
    nf_unregister_net_hook(&init_net, &leap_nf_ops);
err_cdev:
    cdev_del(&leap_cdev);
err_device:
    device_destroy(leap_class, dev_no);
err_class:
    class_destroy(leap_class);
err_chrdev:
    unregister_chrdev_region(dev_no, 1);
err_buffer:
    kfree(leap_buffer);
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

    if (leap_buffer) {
        int i;
        for (i = 0; i < leap_buffer_size; i += PAGE_SIZE) {
            ClearPageReserved(virt_to_page(((unsigned long)leap_buffer) + i));
        }
        kfree(leap_buffer);
    }

    printk(KERN_INFO "LEAP: Module unloaded\n");
}

module_init(leap_init);
module_exit(leap_exit);