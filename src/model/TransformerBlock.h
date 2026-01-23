#ifndef LEAP_TRANSFORMERBLOCK_H
#define LEAP_TRANSFORMERBLOCK_H

#include <torch/torch.h>
#include "ModelArgs.h"
#include "Attention.h"
#include "FeedForward.h"
#include "RMSNorm.h"

namespace Model {
    struct TransformerBlockImpl : torch::nn::Module {
        // Constructor
        TransformerBlockImpl(int64_t layer_id, const ModelArgs &args);

        // Forward pass
        torch::Tensor forward(torch::Tensor x, const torch::Tensor &freqs_cos, const torch::Tensor &freqs_sin);

        // Submodules
        Attention attention{nullptr};
        FeedForward feed_forward{nullptr};
        RMSNorm attention_norm{nullptr};
        RMSNorm ffn_norm{nullptr};

        // Class attributes
        int64_t layer_id;
        int64_t n_heads;
        int64_t dim;
        int64_t head_dim;
    };

    // Wrapper for value semantics
    TORCH_MODULE (TransformerBlock);
} // namespace Model

#endif //LEAP_TRANSFORMERBLOCK_H