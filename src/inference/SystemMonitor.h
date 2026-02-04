#ifndef LEAP_SYSTEM_MONITOR_H
#define LEAP_SYSTEM_MONITOR_H

#include <cstdint>
#include <string>

namespace Inference {

    struct NodeStats {
        float cpu_usage;      // Percentage (0.0 - 100.0)
        float ram_usage;      // Percentage (0.0 - 100.0)
        float temperature;    // Celsius
        uint64_t timestamp;   // Time of measurement
        int split_layer;      // Current Start Layer
        int end_layer;        // Current End Layer
    };

    class SystemMonitor {
    public:
        SystemMonitor();
        ~SystemMonitor() = default;

        NodeStats get_stats();

    private:
        float get_cpu_usage();
        float get_ram_usage();
        float get_temperature();

        // State for CPU calculation (deltas)
        unsigned long long prev_user = 0, prev_nice = 0, prev_system = 0, prev_idle = 0;
        unsigned long long prev_iowait = 0, prev_irq = 0, prev_softirq = 0, prev_steal = 0;
    };

} // namespace Inference

#endif // LEAP_SYSTEM_MONITOR_H