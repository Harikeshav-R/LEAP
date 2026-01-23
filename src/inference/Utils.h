#ifndef LEAP_UTILS_H
#define LEAP_UTILS_H

#include <chrono>
#include <cctype>

namespace Inference {
    class Utils {
    public:
        static long long time_in_ms() {
            const auto now = std::chrono::system_clock::now();
            const auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(now.time_since_epoch());
            return duration.count();
        }

        static unsigned int random_u32(unsigned long long *state) {
            // xorshift rng: https://en.wikipedia.org/wiki/Xorshift#xorshift.2A
            *state ^= *state >> 12;
            *state ^= *state << 25;
            *state ^= *state >> 27;
            return (*state * 0x2545F4914F6CDD1Dull) >> 32;
        }

        static float random_f32(unsigned long long *state) {
            // random float32 in [0,1)
            return (random_u32(state) >> 8) / 16777216.0f;
        }

        static void safe_printf(const char *piece) {
            // piece might be a raw byte token, and we only want to print printable chars or whitespace
            // because some of the other bytes can be various control codes, backspace, etc.
            if (piece == nullptr) {
                return;
            }
            if (piece[0] == '\0') {
                return;
            }
            if (piece[1] == '\0') {
                if (const unsigned char byte_val = piece[0]; !(isprint(byte_val) || isspace(byte_val))) {
                    return; // bad byte, don't print it
                }
            }
            printf("%s", piece);
        }
    };
} // namespace Inference

#endif // LEAP_UTILS_H