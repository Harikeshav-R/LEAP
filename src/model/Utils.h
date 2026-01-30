#ifndef LEAP_UTILS_H
#define LEAP_UTILS_H

#include <torch/torch.h>

namespace Model {
    // Returns a complex tensor of shape [end, dim/2]
    torch::Tensor precompute_freqs_cis(
        int64_t dim,
        int64_t end,
        double theta = 10000.0
    );

    // Applies RoPE using complex number arithmetic
    std::pair<torch::Tensor, torch::Tensor> apply_rotary_emb(
        const torch::Tensor &xq,
        const torch::Tensor &xk,
        const torch::Tensor &freqs_cis
    );
} // namespace Model

#endif //LEAP_UTILS_H