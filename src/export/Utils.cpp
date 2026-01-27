#include "Utils.h"
#include <cmath>
#include <algorithm>
#include <cstring>
#include <iostream>

#if defined(_OPENMP)
#include <omp.h>
#endif

namespace Export {
    void serialize_fp32(std::ofstream &out, const torch::Tensor &tensor) {
        torch::Tensor t = tensor;
        if (t.device().type() != torch::kCPU) t = t.cpu();
        if (t.dtype() != torch::kFloat32) t = t.to(torch::kFloat32);
        if (!t.is_contiguous()) t = t.contiguous();

        const auto data_ptr = t.data_ptr < float > ();
        const auto num_bytes = t.numel() * sizeof(float);
        out.write(reinterpret_cast<const char *>(data_ptr), num_bytes);
    }

    void serialize_int8(std::ofstream &out, const torch::Tensor &tensor) {
        const auto d = tensor.detach().cpu().to(torch::kInt8);
        // Note: .to(kInt8) handles contiguous implicitly if needed or returns copy
        // But to be safe:
        auto contig = d.contiguous();
        const auto data_ptr = contig.data_ptr < int8_t > ();
        const auto num_bytes = contig.numel() * sizeof(int8_t);
        out.write(reinterpret_cast<const char *>(data_ptr), num_bytes);
    }

    // Optimized CPU Quantization implementation
    // Avoids allocating multiple intermediate tensors
    std::tuple<torch::Tensor, torch::Tensor, float> quantize_q80(const torch::Tensor &w_in, int64_t group_size) {
        // Ensure CPU float32 contiguous
        torch::Tensor w = w_in.to(torch::kFloat32).contiguous().cpu();
        const float *w_data = w.data_ptr<float>();
        int64_t numel = w.numel();

        TORCH_CHECK(numel % group_size == 0, "Tensor numel must be divisible by group_size");

        int64_t num_groups = numel / group_size;

        // Output tensors
        auto int8val = torch::empty({numel}, torch::kInt8);
        int8_t *q_data = int8val.data_ptr<int8_t>();

        auto scale = torch::empty({num_groups}, torch::kFloat32);
        float *s_data = scale.data_ptr<float>();

        // Track max error
        float global_max_err = 0.0f;

        // Parallelize over groups
#pragma omp parallel for reduction(max: global_max_err)
        for (int64_t g = 0; g < num_groups; ++g) {
            const float *group_w = w_data + g * group_size;
            int8_t *group_q = q_data + g * group_size;

            // 1. Find max absolute value
            float wmax = 0.0f;
            for (int64_t i = 0; i < group_size; ++i) {
                float val = std::abs(group_w[i]);
                if (val > wmax) wmax = val;
            }

            // 2. Calculate scale
            float s = wmax / 127.0f;
            s_data[g] = s;

            // Avoid division by zero
            float inv_s = (s != 0.0f) ? (1.0f / s) : 0.0f;

            // 3. Quantize and Compute Error
            float max_grp_err = 0.0f;

            for (int64_t i = 0; i < group_size; ++i) {
                float val = group_w[i];
                float q_float = val * inv_s;

                // Round to nearest integer
                int8_t q = static_cast<int8_t>(std::round(q_float));
                group_q[i] = q;

                // Dequantize to check error
                float dq = q * s;
                float err = std::abs(dq - val);
                if (err > max_grp_err) max_grp_err = err;
            }

            if (max_grp_err > global_max_err) {
                global_max_err = max_grp_err;
            }
        }

        // Reshape outputs to match expected return shapes
        // int8val: flattened is fine, or reshape to {num_groups, group_size} ?
        // The original code returned: int8val shaped as {numel} (via .view(-1)) inside logic but logically {num_groups, group_size}
        // Actually original: w.reshape({-1, group_size}) -> int8val same shape.
        // But serialize_int8 flattens it anyway.
        // Let's reshape to match original logic behavior: {num_groups, group_size}
        int8val = int8val.view({num_groups, group_size});

        return std::make_tuple(int8val, scale, global_max_err);
    }
} // namespace Export