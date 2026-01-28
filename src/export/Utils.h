#ifndef LEAP_UTILS_H
#define LEAP_UTILS_H

#include <torch/torch.h>
#include <fstream>
#include <tuple>

namespace Export {
    void serialize_fp32(std::ofstream &out, const torch::Tensor &tensor);

    void serialize_int8(std::ofstream &out, const torch::Tensor &tensor);

    std::tuple<torch::Tensor, torch::Tensor, float> quantize_q80(const torch::Tensor &w_in, int64_t group_size);
} // namespace Export

#endif //LEAP_UTILS_H
