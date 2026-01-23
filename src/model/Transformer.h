#ifndef LEAP_TRANSFORMER_H
#define LEAP_TRANSFORMER_H


#include <torch/torch.h>
#include <vector>
#include <optional>
#include <string>
#include <memory>

#include "ModelArgs.h"
#include "TransformerBlock.h"
#include "RMSNorm.h"

namespace Model {
    class TransformerImpl : public torch::nn::Module {
    public:
        // State
        ModelArgs params;
        torch::optional<torch::Tensor> last_loss;

        // Submodules
        torch::nn::Embedding tok_embeddings{nullptr};
        torch::nn::Dropout dropout{nullptr};
        std::vector<TransformerBlock> layers;
        RMSNorm norm{nullptr};
        torch::nn::Linear output{nullptr};

        // Buffers for RoPE
        torch::Tensor freqs_cos;
        torch::Tensor freqs_sin;

        // Constructor
        explicit TransformerImpl(const ModelArgs &params);

        // Forward Pass
        torch::Tensor forward(const torch::Tensor &tokens,
                              const torch::optional<torch::Tensor> &targets = torch::nullopt);

        // Optimizer Configuration
        std::unique_ptr<torch::optim::AdamW> configure_optimizers(
            double weight_decay,
            double learning_rate,
            std::pair<double, double> betas,
            const std::string &device_type
        ) const;

        // MFU Estimation
        double estimate_mfu(int64_t fwdbwd_per_iter, double dt) const;

        // Generation
        torch::Tensor generate(
            torch::Tensor idx,
            int64_t max_new_tokens,
            double temperature = 1.0,
            std::optional<int64_t> top_k = std::nullopt
        );

    private:
        // Initialization helper
        static void _init_weights(torch::nn::Module &module);
    };

    // Value-semantic wrapper
    TORCH_MODULE (Transformer);
} // namespace Model

#endif // LEAP_TRANSFORMER_H