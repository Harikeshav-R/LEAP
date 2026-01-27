#ifndef LEAP_ATTENTION_H
#define LEAP_ATTENTION_H

#include "ModelArgs.h"
#include <torch/torch.h>
#include <optional>

namespace Model {
    struct KVCache {
        torch::Tensor k; // [B, n_kv_heads, max_seq_len, head_dim]
        torch::Tensor v; // [B, n_kv_heads, max_seq_len, head_dim]
    };

    class AttentionImpl : public torch::nn::Module {
    public:
        explicit AttentionImpl(const ModelArgs &args);

        torch::Tensor forward(
            const torch::Tensor &x,
            const torch::Tensor &freqs_cis,
            int64_t start_pos,
            const std::optional<KVCache> &kv_cache
        );

        torch::nn::Linear wq{nullptr};
        torch::nn::Linear wk{nullptr};
        torch::nn::Linear wv{nullptr};
        torch::nn::Linear wo{nullptr};
        torch::nn::Dropout attn_dropout{nullptr};
        torch::nn::Dropout resid_dropout{nullptr};

    private:
        int64_t n_heads;
        int64_t n_kv_heads;
        int64_t n_rep;
        int64_t head_dim;
        double dropout_prob;
    };

    TORCH_MODULE (Attention);
} // namespace Model

#endif //LEAP_ATTENTION_H