#include "SystemMonitor.h"
#include <fstream>
#include <iostream>
#include <sstream>
#include <vector>
#include <thread>
#include <chrono>

#if defined(__APPLE__)
#include <mach/mach.h>
#include <mach/vm_statistics.h>
#include <mach/mach_types.h>
#include <mach/mach_init.h>
#include <mach/host_info.h>
#include <sys/sysctl.h>
#endif

namespace Inference {

    SystemMonitor::SystemMonitor() {
        // Initialize CPU baseline
        get_cpu_usage();
    }

    NodeStats SystemMonitor::get_stats() {
        NodeStats stats{};
        stats.cpu_usage = get_cpu_usage();
        stats.ram_usage = get_ram_usage();
        stats.temperature = get_temperature();
        stats.timestamp = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::system_clock::now().time_since_epoch()).count();
        return stats;
    }

#if defined(__linux__)
    float SystemMonitor::get_cpu_usage() {
        std::ifstream file("/proc/stat");
        std::string line;
        if (!std::getline(file, line)) return 0.0f;

        std::istringstream iss(line);
        std::string cpu_label;
        unsigned long long user, nice, system, idle, iowait, irq, softirq, steal;

        iss >> cpu_label >> user >> nice >> system >> idle >> iowait >> irq >> softirq >> steal;

        unsigned long long total_idle = idle + iowait;
        unsigned long long total_non_idle = user + nice + system + irq + softirq + steal;
        unsigned long long total = total_idle + total_non_idle;

        unsigned long long prev_total_idle = prev_idle + prev_iowait;
        unsigned long long prev_total_non_idle = prev_user + prev_nice + prev_system + prev_irq + prev_softirq + prev_steal;
        unsigned long long prev_total = prev_total_idle + prev_total_non_idle;

        float percentage = 0.0f;
        unsigned long long total_d = total - prev_total;
        unsigned long long idle_d = total_idle - prev_total_idle;

        if (total_d > 0) {
            percentage = (float)(total_d - idle_d) / total_d * 100.0f;
        }

        prev_user = user; prev_nice = nice; prev_system = system; prev_idle = idle;
        prev_iowait = iowait; prev_irq = irq; prev_softirq = softirq; prev_steal = steal;

        return percentage;
    }

    float SystemMonitor::get_ram_usage() {
        std::ifstream file("/proc/meminfo");
        std::string line;
        unsigned long long total_mem = 0;
        unsigned long long free_mem = 0;
        unsigned long long buffers = 0;
        unsigned long long cached = 0;

        while (std::getline(file, line)) {
            std::istringstream iss(line);
            std::string key;
            unsigned long long val;
            std::string unit;
            iss >> key >> val >> unit;

            if (key == "MemTotal:") total_mem = val;
            else if (key == "MemFree:") free_mem = val;
            else if (key == "Buffers:") buffers = val;
            else if (key == "Cached:") cached = val;
        }

        if (total_mem == 0) return 0.0f;

        // Linux 'free' is actually free + buffers + cached
        unsigned long long actual_free = free_mem + buffers + cached;
        unsigned long long used = total_mem - actual_free;

        return (float)used / total_mem * 100.0f;
    }

    float SystemMonitor::get_temperature() {
        // Attempt to read from thermal zone 0
        std::ifstream file("/sys/class/thermal/thermal_zone0/temp");
        if (file) {
            int temp;
            file >> temp;
            return (float)temp / 1000.0f;
        }
        return 0.0f; // Sensor not found
    }

#elif defined(__APPLE__)
    float SystemMonitor::get_cpu_usage() {
        host_cpu_load_info_data_t cpuinfo;
        mach_msg_type_number_t count = HOST_CPU_LOAD_INFO_COUNT;
        if (host_statistics(mach_host_self(), HOST_CPU_LOAD_INFO, (host_info_t)&cpuinfo, &count) == KERN_SUCCESS) {
            unsigned long long user = cpuinfo.cpu_ticks[CPU_STATE_USER];
            unsigned long long nice = cpuinfo.cpu_ticks[CPU_STATE_NICE];
            unsigned long long system = cpuinfo.cpu_ticks[CPU_STATE_SYSTEM];
            unsigned long long idle = cpuinfo.cpu_ticks[CPU_STATE_IDLE];

            unsigned long long total = user + nice + system + idle;
            unsigned long long prev_total = prev_user + prev_nice + prev_system + prev_idle;

            float percentage = 0.0f;
            unsigned long long total_d = total - prev_total;
            unsigned long long idle_d = idle - prev_idle;

            if (total_d > 0) {
                 percentage = (float)(total_d - idle_d) / total_d * 100.0f;
            }

            prev_user = user; prev_nice = nice; prev_system = system; prev_idle = idle;
            return percentage;
        }
        return 0.0f;
    }

    float SystemMonitor::get_ram_usage() {
        vm_size_t page_size;
        mach_port_t mach_port;
        mach_msg_type_number_t count;
        vm_statistics64_data_t vm_stats;

        mach_port = mach_host_self();
        count = HOST_VM_INFO64_COUNT;
        if (host_page_size(mach_port, &page_size) == KERN_SUCCESS &&
            host_statistics64(mach_port, HOST_VM_INFO64, (host_info_t)&vm_stats, &count) == KERN_SUCCESS) {
            
            long long free_memory = (int64_t)vm_stats.free_count * page_size;
            long long used_memory = ((int64_t)vm_stats.active_count +
                                     (int64_t)vm_stats.inactive_count +
                                     (int64_t)vm_stats.wire_count) *  page_size;
            
            long long total = free_memory + used_memory;
            if (total > 0)
                return (float)used_memory / total * 100.0f;
        }
        return 0.0f;
    }

    float SystemMonitor::get_temperature() {
        // macOS temperature requires IOKit/SMC which is complex. Returning dummy or 0.
        // For development safety, we can return a safe value or 0.
        return 45.0f; // Mock value
    }
#else
    // Fallback for Windows/Other
    float SystemMonitor::get_cpu_usage() { return 0.0f; }
    float SystemMonitor::get_ram_usage() { return 0.0f; }
    float SystemMonitor::get_temperature() { return 0.0f; }
#endif

} // namespace Inference
