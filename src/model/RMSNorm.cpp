#include "RMSNorm.h"

namespace Model {
    RMSNormImpl::RMSNormImpl(int64_t dim, const double eps)
        : eps(eps) {
        weight = register_parameter("weight", torch::ones({dim}));
    }

    torch::Tensor RMSNormImpl::forward(const torch::Tensor &x) const {
        // Cast to float32 for stability
        auto x_float = x.to(torch::kFloat32);
        
        // Optimization: Use linalg_vector_norm to compute sqrt(sum(x^2)) without materializing x^2 tensor
        // mean(x^2) = norm(x)^2 / N
        auto norm_x = torch::linalg_vector_norm(x_float, 2, {-1}, true);
        auto mean_sq = norm_x.pow_(2) / x_float.size(-1);
        
        auto rstd = torch::rsqrt(mean_sq + eps);
        
        // Convert back to input type and scale
        return (x_float * rstd).type_as(x) * weight;
    }
} // namespace Model
