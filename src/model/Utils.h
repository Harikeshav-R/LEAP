#ifndef LEAP_UTILS_H
#define LEAP_UTILS_H

#include <cstdint>
#include <torch/torch.h>

namespace Model {
    std::pair<torch::Tensor, torch::Tensor> precompute_freqs_cis(
        int64_t dim,
        int64_t end,
        double theta = 10000.0
    );

    torch::Tensor reshape_for_broadcast(const torch::Tensor &freqs_cis, const torch::Tensor &x);

    std::pair<torch::Tensor, torch::Tensor> apply_rotary_emb(
        const torch::Tensor &xq,
        const torch::Tensor &xk,
        const torch::Tensor &freqs_cos,
        const torch::Tensor &freqs_sin
    );

    torch::Tensor repeat_kv(const torch::Tensor &x, int64_t n_rep);
} // namespace Models

#endif //LEAP_UTILS_H