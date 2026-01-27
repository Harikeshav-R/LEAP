#ifndef LEAP_TRANSFORMERBLOCK_H
#define LEAP_TRANSFORMERBLOCK_H

#include <torch/torch.h>
#include "ModelArgs.h"
#include "Attention.h"
#include "FeedForward.h"
#include "RMSNorm.h"

namespace Model {
    struct TransformerBlockImpl : torch::nn::Module {
        TransformerBlockImpl(int64_t layer_id, const ModelArgs &args);

        torch::Tensor forward(
            torch::Tensor x,
            const torch::Tensor &freqs_cis,
            int64_t start_pos,
            const std::optional<KVCache> &kv_cache
        );

        Attention attention{nullptr};
        FeedForward feed_forward{nullptr};
        RMSNorm attention_norm{nullptr};
        RMSNorm ffn_norm{nullptr};

        int64_t layer_id;
        int64_t n_heads;
        int64_t dim;
        int64_t head_dim;
    };

    TORCH_MODULE (TransformerBlock);
} // namespace Model

#endif //LEAP_TRANSFORMERBLOCK_H