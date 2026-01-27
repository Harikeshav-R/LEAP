#include "Transformer.h"
#include "FloatTransformer.h"
#include "QuantizedTransformer.h"
#include <fstream>
#include <fcntl.h>
#include <unistd.h>
#include <sys/mman.h>
#include <stdexcept>
#include <iostream>
#include <filesystem>
#include <cmath>

namespace Inference {
    // RAII wrapper for mmap to ensure safety if Transformer constructor throws
    class ScopedMmap {
    public:
        int fd = -1;
        void *ptr = MAP_FAILED;
        size_t size = 0;

        explicit ScopedMmap(const std::string &path) {
            size = std::filesystem::file_size(path);
            fd = open(path.c_str(), O_RDONLY);
            if (fd == -1) {
                throw std::runtime_error("open failed!");
            }
            ptr = mmap(nullptr, size, PROT_READ, MAP_PRIVATE, fd, 0);
            if (ptr == MAP_FAILED) {
                close(fd);
                fd = -1;
                throw std::runtime_error("mmap failed!");
            }
#if defined(__linux__) || defined(__APPLE__)
            // Optimize memory access patterns
            // MADV_SEQUENTIAL: Expect sequential page access (good for loading weights)
            // MADV_WILLNEED: Notify OS that we will need these pages soon (triggers readahead)
            madvise(ptr, size, MADV_SEQUENTIAL);
            madvise(ptr, size, MADV_WILLNEED);
#endif
        }

        ~ScopedMmap() {
            if (ptr != MAP_FAILED) {
                munmap(ptr, size);
            }
            if (fd != -1) {
                close(fd);
            }
        }

        void release() {
            fd = -1;
            ptr = MAP_FAILED;
        }
    };

    std::unique_ptr<Transformer> Transformer::create(const std::string &checkpoint_path) {
        std::ifstream file(checkpoint_path, std::ios::binary);
        if (!file.is_open()) {
            throw std::runtime_error("Couldn't open file " + checkpoint_path);
        }

        int magic;
        if (!file.read(reinterpret_cast<char *>(&magic), sizeof(int))) {
            throw std::runtime_error("Failed to read magic number");
        }

        int version = 0;
        Config config{};
        int shared_weights = 0;
        int group_size = 0;
        size_t header_size = 0;
        bool is_quantized = false;

        if (magic == 0x616B3432) {
            if (!file.read(reinterpret_cast<char *>(&version), sizeof(int))) {
                throw std::runtime_error("Failed to read version");
            }

            if (version == 1) {
                if (!file.read(reinterpret_cast<char *>(&config), sizeof(Config))) {
                    throw std::runtime_error("Failed to read config");
                }
                header_size = 256;
            } else if (version == 2) {
                is_quantized = true;
                header_size = 256;
                if (!file.read(reinterpret_cast<char *>(&config), sizeof(Config))) {
                    throw std::runtime_error("Failed to read config");
                }
                uint8_t shared_classifier_byte;
                if (!file.read(reinterpret_cast<char *>(&shared_classifier_byte), sizeof(uint8_t))) {
                    throw std::runtime_error("Failed to read shared_classifier");
                }
                shared_weights = shared_classifier_byte;
                if (!file.read(reinterpret_cast<char *>(&group_size), sizeof(int))) {
                    throw std::runtime_error("Failed to read group_size");
                }
            } else {
                throw std::runtime_error("Bad file version " + std::to_string(version));
            }
        } else {
            // Legacy Version 0: Float
            file.seekg(0, std::ios::beg);
            if (!file.read(reinterpret_cast<char *>(&config), sizeof(Config))) {
                throw std::runtime_error("Failed to read config");
            }
            header_size = sizeof(Config);
        }

        file.close();

        if (!is_quantized) {
            shared_weights = config.vocab_size > 0 ? 1 : 0;
            config.vocab_size = std::abs(config.vocab_size);
        }

        ScopedMmap mmap_resource(checkpoint_path);
        void *weights_ptr = static_cast<char *>(mmap_resource.ptr) + header_size;

        std::unique_ptr<Transformer> t;
        if (is_quantized) {
            t = std::make_unique<QuantizedTransformer>(config, mmap_resource.fd, mmap_resource.ptr, mmap_resource.size,
                                                       weights_ptr, shared_weights, group_size);
        } else {
            t = std::make_unique<FloatTransformer>(config, mmap_resource.fd, mmap_resource.ptr, mmap_resource.size,
                                                   static_cast<float *>(weights_ptr),
                                                   shared_weights);
        }

        mmap_resource.release();
        return t;
    }
} // namespace Inference