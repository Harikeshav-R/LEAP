#ifndef LEAP_ATTENTION_H
#define LEAP_ATTENTION_H

#include "ModelArgs.h"


#include <torch/torch.h>
#include <optional>
#include <vector>


namespace Model {
    // --- Attention Module ---
    class AttentionImpl : public torch::nn::Module {
    public:
        explicit AttentionImpl(const ModelArgs &args);

        torch::Tensor forward(
            const torch::Tensor &x,
            const torch::Tensor &freqs_cos,
            const torch::Tensor &freqs_sin
        );

        // Submodules
        torch::nn::Linear wq{nullptr};
        torch::nn::Linear wk{nullptr};
        torch::nn::Linear wv{nullptr};
        torch::nn::Linear wo{nullptr};
        torch::nn::Dropout attn_dropout{nullptr};
        torch::nn::Dropout resid_dropout{nullptr};

    private:
        int n_heads;
        int n_kv_heads;
        int n_local_heads;
        int n_local_kv_heads;
        int n_rep;
        int head_dim;
        float dropout_prob;

        // Buffer
        torch::Tensor mask;
    };

    // Use TORCH_MODULE to create a value-semantic wrapper (Attention)
    // that points to the implementation (AttentionImpl).
    TORCH_MODULE (Attention);
} // namespace Model


#endif //LEAP_ATTENTION_H