#include "Utils.h"

namespace Model {
    torch::Tensor precompute_freqs_cis(
        const int64_t dim,
        const int64_t end,
        const double theta
    ) {
        // freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
        auto freqs = 1.0 / torch::pow(theta, torch::arange(0, dim, 2, torch::kFloat) / static_cast<double>(dim));

        // t = torch.arange(end, device=freqs.device)
        const auto t = torch::arange(end, freqs.options());

        // freqs = torch.outer(t, freqs).float()
        freqs = torch::outer(t, freqs);

        // Turn into complex numbers: cos(freqs) + i*sin(freqs)
        return torch::polar(torch::ones_like(freqs), freqs);
    }

    std::pair<torch::Tensor, torch::Tensor> apply_rotary_emb(
        const torch::Tensor &xq,
        const torch::Tensor &xk,
        const torch::Tensor &freqs_cis
    ) {
        // xq shape: [bs, seqlen, n_heads, head_dim]
        // freqs_cis shape: [seqlen, head_dim/2]

        // Reshape xq/xk to [bs, seqlen, n_heads, head_dim/2, 2] and view as complex
        // We use view_as_complex which is a zero-copy cast
        const auto xq_c = torch::view_as_complex(xq.to(torch::kFloat).reshape({
            xq.size(0), xq.size(1), xq.size(2), -1, 2
        }));
        const auto xk_c = torch::view_as_complex(xk.to(torch::kFloat).reshape({
            xk.size(0), xk.size(1), xk.size(2), -1, 2
        }));

        // Reshape freqs_cis for broadcasting: [1, seqlen, 1, head_dim/2]
        const auto freqs = freqs_cis.view({1, xq.size(1), 1, xq.size(3) / 2});

        // Perform rotation in complex domain
        // (a + ib) * (c + id) = (ac - bd) + i(ad + bc)
        // This handles the rotation mathematically equivalently to the matrix multiplication
        const auto xq_out_c = xq_c * freqs;
        const auto xk_out_c = xk_c * freqs;

        // Convert back to real and flatten the last two dimensions
        const auto xq_out = torch::view_as_real(xq_out_c).flatten(3);
        const auto xk_out = torch::view_as_real(xk_out_c).flatten(3);

        return {xq_out.type_as(xq), xk_out.type_as(xk)};
    }
} // namespace Model
