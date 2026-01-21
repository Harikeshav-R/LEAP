#include "Utils.h"

namespace Model {
    std::pair<torch::Tensor, torch::Tensor> precompute_freqs_cis(
        const int64_t dim,
        const int64_t end,
        const double theta
    ) {
        // freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
        // Note: slice [: (dim // 2)] is implied by arange(0, dim, 2)
        auto freqs = 1.0 / torch::pow(theta, torch::arange(0, dim, 2).to(torch::kFloat) / static_cast<double>(dim));

        // t = torch.arange(end, device=freqs.device)
        // We use freqs.options() to copy device and dtype settings
        const auto t = torch::arange(end, freqs.options());

        // freqs = torch.outer(t, freqs).float()
        freqs = torch::outer(t, freqs).to(torch::kFloat);

        // freqs_cos = torch.cos(freqs)
        // freqs_sin = torch.sin(freqs)
        auto freqs_cos = torch::cos(freqs);
        auto freqs_sin = torch::sin(freqs);

        return {freqs_cos, freqs_sin};
    }

    torch::Tensor reshape_for_broadcast(const torch::Tensor &freqs_cis, const torch::Tensor &x) {
        const int64_t ndim = x.dim();

        // assert 0 <= 1 < ndim
        TORCH_CHECK(1 < ndim, "ndim must be greater than 1");

        // assert freqs_cis.shape == (x.shape[1], x.shape[-1])
        TORCH_CHECK(freqs_cis.size(0) == x.size(1) && freqs_cis.size(1) == x.size(-1),
                    "freqs_cis shape mismatch with x dimensions");

        // shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
        std::vector<int64_t> shape;
        shape.reserve(ndim);

        for (int64_t i = 0; i < ndim; ++i) {
            if (i == 1 || i == ndim - 1) {
                shape.push_back(x.size(i));
            } else {
                shape.push_back(1);
            }
        }

        return freqs_cis.view(shape);
    }

    std::pair<torch::Tensor, torch::Tensor> apply_rotary_emb(
        const torch::Tensor &xq,
        const torch::Tensor &xk,
        const torch::Tensor &freqs_cos,
        const torch::Tensor &freqs_sin
    ) {
        // xq_r, xq_i = xq.float().reshape(xq.shape[:-1] + (-1, 2)).unbind(-1)

        // Construct the new shape for xq manually: (bs, slen, n_heads, head_dim/2, 2)
        auto xq_shape = xq.sizes().vec();
        xq_shape.pop_back();
        xq_shape.push_back(-1);
        xq_shape.push_back(2);

        const auto xq_reshaped = xq.to(torch::kFloat).reshape(xq_shape);
        const auto xq_unbound = xq_reshaped.unbind(-1);
        const auto &xq_r = xq_unbound[0];
        const auto &xq_i = xq_unbound[1];

        // Construct the new shape for xk manually
        auto xk_shape = xk.sizes().vec();
        xk_shape.pop_back();
        xk_shape.push_back(-1);
        xk_shape.push_back(2);

        const auto xk_reshaped = xk.to(torch::kFloat).reshape(xk_shape);
        const auto xk_unbound = xk_reshaped.unbind(-1);
        const auto &xk_r = xk_unbound[0];
        const auto &xk_i = xk_unbound[1];

        // reshape freqs_cos and freqs_sin for broadcasting
        const auto cos_broadcast = reshape_for_broadcast(freqs_cos, xq_r);
        const auto sin_broadcast = reshape_for_broadcast(freqs_sin, xq_r);

        // apply rotation using real numbers
        auto xq_out_r = xq_r * cos_broadcast - xq_i * sin_broadcast;
        auto xq_out_i = xq_r * sin_broadcast + xq_i * cos_broadcast;
        auto xk_out_r = xk_r * cos_broadcast - xk_i * sin_broadcast;
        auto xk_out_i = xk_r * sin_broadcast + xk_i * cos_broadcast;

        // flatten last two dimensions
        // xq_out = torch.stack([xq_out_r, xq_out_i], dim=-1).flatten(3)
        const auto xq_out = torch::stack({xq_out_r, xq_out_i}, -1).flatten(3);
        const auto xk_out = torch::stack({xk_out_r, xk_out_i}, -1).flatten(3);

        return {xq_out.type_as(xq), xk_out.type_as(xk)};
    }

    torch::Tensor repeat_kv(const torch::Tensor &x, int64_t n_rep) {
        // bs, slen, n_kv_heads, head_dim = x.shape
        int64_t bs = x.size(0);
        int64_t slen = x.size(1);
        int64_t n_kv_heads = x.size(2);
        int64_t head_dim = x.size(3);

        if (n_rep == 1) {
            return x;
        }

        // x[:, :, :, None, :]
        // .expand(bs, slen, n_kv_heads, n_rep, head_dim)
        // .reshape(bs, slen, n_kv_heads * n_rep, head_dim)

        return x.unsqueeze(3)
                .expand({bs, slen, n_kv_heads, n_rep, head_dim})
                .reshape({bs, slen, n_kv_heads * n_rep, head_dim});
    }
} // namespace Models
