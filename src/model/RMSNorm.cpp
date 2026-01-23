#include "RMSNorm.h"

namespace Model {
    RMSNormImpl::RMSNormImpl(int64_t dim, const double eps)
        : eps(eps) {
        // Register the parameter so the optimizer can find it
        weight = register_parameter("weight", torch::ones({dim}));
    }

    torch::Tensor RMSNormImpl::_norm(const torch::Tensor &x) const {
        auto mean_sq = x.pow(2).mean(/*dim=*/{-1}, /*keepdim=*/true);
        return x * torch::rsqrt(mean_sq + eps);
    }

    torch::Tensor RMSNormImpl::forward(const torch::Tensor &x) const {
        const auto x_float = x.to(torch::kFloat32);
        const auto normed = _norm(x_float);
        const auto output = normed.type_as(x);
        return output * weight;
    }
} // namespace Model