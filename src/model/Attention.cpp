#include "Attention.h"
#include "Utils.h"


namespace Model {
    AttentionImpl::AttentionImpl(const ModelArgs &args) {
        n_heads = args.n_heads;
        n_kv_heads = args.n_kv_heads.has_value() ? args.n_kv_heads.value() : args.n_heads;

        TORCH_CHECK(args.n_heads % n_kv_heads == 0, "n_heads must be divisible by n_kv_heads");

        constexpr int model_parallel_size = 1;
        n_local_heads = n_heads / model_parallel_size;
        n_local_kv_heads = n_kv_heads / model_parallel_size;
        n_rep = n_local_heads / n_local_kv_heads;
        head_dim = args.dim / args.n_heads;
        dropout_prob = args.dropout;

        // Register Linear layers
        // Note: Python nn.Linear defaults are bias=True, but code specifies bias=False
        wq = register_module(
            "wq", torch::nn::Linear(torch::nn::LinearOptions(args.dim, n_heads * head_dim).bias(false)));
        wk = register_module(
            "wk", torch::nn::Linear(torch::nn::LinearOptions(args.dim, n_kv_heads * head_dim).bias(false)));
        wv = register_module(
            "wv", torch::nn::Linear(torch::nn::LinearOptions(args.dim, n_kv_heads * head_dim).bias(false)));
        wo = register_module(
            "wo", torch::nn::Linear(torch::nn::LinearOptions(n_heads * head_dim, args.dim).bias(false)));

        // Register Dropout layers
        attn_dropout = register_module("attn_dropout", torch::nn::Dropout(args.dropout));
        resid_dropout = register_module("resid_dropout", torch::nn::Dropout(args.dropout));
    }

    torch::Tensor AttentionImpl::forward(
        const torch::Tensor &x,
        const torch::Tensor &freqs_cos,
        const torch::Tensor &freqs_sin
    ) {
        auto bsz = x.size(0);
        auto seqlen = x.size(1);

        // QKV
        torch::Tensor xq = wq->forward(x);
        torch::Tensor xk = wk->forward(x);
        torch::Tensor xv = wv->forward(x);

        // View reshape
        xq = xq.view({bsz, seqlen, n_local_heads, head_dim});
        xk = xk.view({bsz, seqlen, n_local_kv_heads, head_dim});
        xv = xv.view({bsz, seqlen, n_local_kv_heads, head_dim});

        // RoPE relative positional embeddings
        // We use std::tie to unpack the pair returned by the external function
        std::tie(xq, xk) = apply_rotary_emb(xq, xk, freqs_cos, freqs_sin);

        // Grouped multiquery attention: expand out keys and values
        xk = repeat_kv(xk, n_rep); // (bs, seqlen, n_local_heads, head_dim)
        xv = repeat_kv(xv, n_rep); // (bs, seqlen, n_local_heads, head_dim)

        // Make heads into a batch dimension
        xq = xq.transpose(1, 2); // (bs, n_local_heads, seqlen, head_dim)
        xk = xk.transpose(1, 2);
        xv = xv.transpose(1, 2);

        torch::Tensor output = at::scaled_dot_product_attention(
            xq,
            xk,
            xv,
            std::nullopt,
            is_training() ? dropout_prob : 0.0,
            true
        );

        // Restore time as batch dimension and concat heads
        // output = output.transpose(1, 2).contiguous().view(bsz, seqlen, -1)
        output = output.transpose(1, 2).contiguous().view({bsz, seqlen, -1});

        // Final projection into the residual stream
        output = wo->forward(output);
        output = resid_dropout->forward(output);

        return output;
    }
} // namespace Model
