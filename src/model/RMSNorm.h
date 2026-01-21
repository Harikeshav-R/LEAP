#ifndef LEAP_RMSNORM_H
#define LEAP_RMSNORM_H

#include <torch/torch.h>

namespace Model {
    struct RMSNormImpl : torch::nn::Module {
        double eps;
        torch::Tensor weight;

        RMSNormImpl(int64_t dim, double eps);

        torch::Tensor _norm(const torch::Tensor &x) const;

        torch::Tensor forward(const torch::Tensor &x) const;
    };

    // 2. The Wrapper Macro
    // This creates the class 'Model::RMSNorm'
    TORCH_MODULE(RMSNorm);
} // namespace Model

#endif //LEAP_RMSNORM_H
