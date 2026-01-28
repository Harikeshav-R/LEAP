#include "TransformerBlock.h"

namespace Model {
    TransformerBlockImpl::TransformerBlockImpl(const int64_t layer_id, const ModelArgs &args)
        : layer_id(layer_id),
          n_heads(args.n_heads),
          dim(args.dim),
          head_dim(args.dim / args.n_heads) {
        attention = register_module("attention", Attention(args));

        int64_t hidden_dim_val = args.hidden_dim.value_or(-1);
        feed_forward = register_module("feed_forward", FeedForward(
                                           args.dim,
                                           hidden_dim_val,
                                           args.multiple_of,
                                           args.dropout
                                       ));

        attention_norm = register_module("attention_norm", RMSNorm(args.dim, args.norm_eps));
        ffn_norm = register_module("ffn_norm", RMSNorm(args.dim, args.norm_eps));
    }

    torch::Tensor TransformerBlockImpl::forward(
        torch::Tensor x,
        const torch::Tensor &freqs_cis,
        const int64_t start_pos,
        const std::optional<KVCache> &kv_cache
    ) {
        x.add_(attention->forward(attention_norm->forward(x), freqs_cis, start_pos, kv_cache));
        x.add_(feed_forward->forward(ffn_norm->forward(x)));
        return x;
    }
} // namespace Model
