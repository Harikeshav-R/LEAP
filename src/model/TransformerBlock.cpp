#include "TransformerBlock.h"

namespace Model {
    TransformerBlockImpl::TransformerBlockImpl(const int64_t layer_id, const ModelArgs &args)
        : layer_id(layer_id),
          n_heads(args.n_heads),
          dim(args.dim),
          head_dim(args.dim / args.n_heads) {
        // 1. Initialize Attention
        // Pass the full args struct as required by AttentionImpl
        attention = register_module("attention", Attention(args));

        // 2. Initialize FeedForward
        // We must convert the std::optional<int> hidden_dim to int64_t.
        // If it doesn't have a value, we pass -1 (as per FeedForwardImpl comments).
        int64_t hidden_dim_val = args.hidden_dim.value_or(-1);

        feed_forward = register_module("feed_forward", FeedForward(
                                           args.dim,
                                           hidden_dim_val,
                                           args.multiple_of,
                                           args.dropout
                                       ));

        // 3. Initialize RMSNorm layers
        attention_norm = register_module("attention_norm", RMSNorm(args.dim, args.norm_eps));
        ffn_norm = register_module("ffn_norm", RMSNorm(args.dim, args.norm_eps));
    }

    torch::Tensor TransformerBlockImpl::forward(torch::Tensor x, const torch::Tensor &freqs_cos,
                                                const torch::Tensor &freqs_sin) {
        // Python: h = x + self.attention.forward(self.attention_norm(x), freqs_cos, freqs_sin)
        const auto h = x + attention->forward(attention_norm->forward(x), freqs_cos, freqs_sin);

        // Python: out = h + self.feed_forward.forward(self.ffn_norm(h))
        auto out = h + feed_forward->forward(ffn_norm->forward(h));

        return out;
    }
} // namespace Model