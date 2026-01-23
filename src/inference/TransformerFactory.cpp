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

namespace Inference {
    std::unique_ptr<Transformer> Transformer::create(const std::string &checkpoint_path) {
        std::ifstream file(checkpoint_path, std::ios::binary);
        if (!file.is_open()) {
            throw std::runtime_error("Couldn't open file " + checkpoint_path);
        }

        // Attempt to read the magic number first
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
                // Version 1: Float
                if (!file.read(reinterpret_cast<char *>(&config), sizeof(Config))) {
                    throw std::runtime_error("Failed to read config");
                }
                header_size = 256;
            } else if (version == 2) {
                // Version 2: Quantized
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
            // The "magic" int we read was actually config.dim. Rewind.
            file.seekg(0, std::ios::beg);
            if (!file.read(reinterpret_cast<char *>(&config), sizeof(Config))) {
                throw std::runtime_error("Failed to read config");
            }
            header_size = sizeof(Config);
        }

        file.close();

        // Handle negative vocab_size hack for unshared weights (Legacy/V1)
        if (!is_quantized) {
            shared_weights = config.vocab_size > 0 ? 1 : 0;
            config.vocab_size = std::abs(config.vocab_size);
        }

        // Get file size
        size_t file_size = std::filesystem::file_size(checkpoint_path);

        // mmap
        int fd = open(checkpoint_path.c_str(), O_RDONLY);
        if (fd == -1) {
            throw std::runtime_error("open failed!");
        }

        void *mmap_ptr = mmap(nullptr, file_size, PROT_READ, MAP_PRIVATE, fd, 0);
        if (mmap_ptr == MAP_FAILED) {
            close(fd);
            throw std::runtime_error("mmap failed!");
        }

        void *weights_ptr = static_cast<char *>(mmap_ptr) + header_size;

        if (is_quantized) {
            return std::make_unique<QuantizedTransformer>(config, fd, mmap_ptr, file_size, weights_ptr, shared_weights,
                                                          group_size);
        } else {
            return std::make_unique<FloatTransformer>(config, fd, mmap_ptr, file_size,
                                                      static_cast<float *>(weights_ptr),
                                                      shared_weights);
        }
    }
} // namespace Inference