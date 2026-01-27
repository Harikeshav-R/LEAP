#include "Attention.h"
#include "Utils.h"

namespace Model {
    AttentionImpl::AttentionImpl(const ModelArgs &args) {
        n_heads = args.n_heads;
        n_kv_heads = args.n_kv_heads.value_or(args.n_heads);
        n_rep = n_heads / n_kv_heads;
        head_dim = args.dim / n_heads;
        dropout_prob = args.dropout;

        wq = register_module("wq", torch::nn::Linear(torch::nn::LinearOptions(args.dim, n_heads * head_dim).bias(false)));
        wk = register_module("wk", torch::nn::Linear(torch::nn::LinearOptions(args.dim, n_kv_heads * head_dim).bias(false)));
        wv = register_module("wv", torch::nn::Linear(torch::nn::LinearOptions(args.dim, n_kv_heads * head_dim).bias(false)));
        wo = register_module("wo", torch::nn::Linear(torch::nn::LinearOptions(n_heads * head_dim, args.dim).bias(false)));

        attn_dropout = register_module("attn_dropout", torch::nn::Dropout(args.dropout));
        resid_dropout = register_module("resid_dropout", torch::nn::Dropout(args.dropout));
    }

    torch::Tensor AttentionImpl::forward(
        const torch::Tensor &x,
        const torch::Tensor &freqs_cis,
        const int64_t start_pos,
        const std::optional<KVCache> &kv_cache
    ) {
        const auto bsz = x.size(0);
        const auto seqlen = x.size(1);

        auto xq = wq->forward(x).view({bsz, seqlen, n_heads, head_dim});
        auto xk = wk->forward(x).view({bsz, seqlen, n_kv_heads, head_dim});
        auto xv = wv->forward(x).view({bsz, seqlen, n_kv_heads, head_dim});

        std::tie(xq, xk) = apply_rotary_emb(xq, xk, freqs_cis);

        // KV Cache Management
        torch::Tensor keys, values;
        
        if (kv_cache.has_value()) {
            // Transpose to [B, n_kv, S, D] for storage
            auto xk_t = xk.transpose(1, 2);
            auto xv_t = xv.transpose(1, 2);

            // Update cache in-place
            // cache.k[:, :, start_pos : start_pos+seqlen, :] = xk_t
            // We use .copy_() for safe in-place update
            kv_cache->k.slice(2, start_pos, start_pos + seqlen).copy_(xk_t);
            kv_cache->v.slice(2, start_pos, start_pos + seqlen).copy_(xv_t);

            // Retrieve full history for attention: [0 ... start_pos + seqlen]
            keys = kv_cache->k.slice(2, 0, start_pos + seqlen);
            values = kv_cache->v.slice(2, 0, start_pos + seqlen);
        } else {
            // No cache (training or simple inference), use current only
            keys = xk.transpose(1, 2);
            values = xv.transpose(1, 2);
        }

        // Grouped Query Attention: Zero-Copy Broadcasting
        // We reshape Q to [B, n_kv_heads, n_rep, S, D]
        // We reshape K, V to [B, n_kv_heads, 1, S, D]
        // SDPA will treat the first 3 dims as batch and broadcast the 1 to n_rep automatically.
        
        // xq comes in as [B, n_heads, S, D]. Transpose to [B, S, n_heads, D] was planned, 
        // but we need to align with keys first.
        
        // Current shapes:
        // xq: [B, n_heads, S, D] -> View as [B, n_kv_heads, n_rep, S, D]
        // keys: [B, n_kv_heads, S, D] -> View as [B, n_kv_heads, 1, S, D]
        
        auto xq_view = xq.view({bsz, n_kv_heads, n_rep, seqlen, head_dim});
        auto keys_view = keys.view({bsz, n_kv_heads, 1, seqlen, head_dim});
        auto values_view = values.view({bsz, n_kv_heads, 1, seqlen, head_dim});

        // Flash Attention / Scaled Dot Product Attention
        // If seqlen > 1, we are in prefill (processing prompt), so we need a causal mask.
        // If seqlen == 1, we are in decoding (generating one token), attending to all past keys (kv_cache).
        bool is_causal = seqlen > 1;

        auto output = at::scaled_dot_product_attention(
            xq_view,
            keys_view,
            values_view,
            std::nullopt,
            is_training() ? dropout_prob : 0.0,
            is_causal
        );

        // Output is [B, n_kv_heads, n_rep, S, D]
        // We need [B, S, n_heads, D] effectively, but usually we go to [B, S, H*D] eventually.
        // Flatten head dims: [B, n_heads, S, D]
        output = output.view({bsz, n_heads, seqlen, head_dim});

        // Transpose to [B, S, n_heads, D] for final view
        output = output.transpose(1, 2).contiguous().view({bsz, seqlen, -1});
        return resid_dropout->forward(wo->forward(output));
    }
} // namespace Model