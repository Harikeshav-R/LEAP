#ifndef LEAP_EXPORT_H
#define LEAP_EXPORT_H

#include <string>
#include <model/Transformer.h>

namespace Export {
    /**
     * Export the model weights in full float32 .bin file to be read from C.
     */
    void float32_export(const Model::Transformer &model, const std::string &filepath);

    /**
     * Export the model weights in Q8_0 into .bin file to be read from C.
     * - Quantize all weights to symmetric int8, in range [-127, 127]
     * - RMSNorm params are kept in fp32
     */
    void int8_export(const Model::Transformer &model, const std::string &filepath, int64_t group_size = 64);
}
#endif //LEAP_EXPORT_H