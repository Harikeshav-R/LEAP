#ifndef LEAP_TRANSFORMER_H
#define LEAP_TRANSFORMER_H

#include "Config.h"
#include <string>
#include <memory>

namespace Inference {
    class Transformer {
    public:
        Config config{};

        virtual ~Transformer() = default;

        // The core function: forward pass
        virtual float *forward(int token, int pos) = 0;

        // Factory method to create the appropriate Transformer (Float or Quantized) based on file
        static std::unique_ptr<Transformer> create(const std::string &checkpoint_path);
    };
} // namespace Inference


#endif // LEAP_TRANSFORMER_H