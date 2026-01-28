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
        ModelArgs params;
        torch::optional<torch::Tensor> last_loss;

        torch::nn::Embedding tok_embeddings{nullptr};
        torch::nn::Dropout dropout{nullptr};
        std::vector<TransformerBlock> layers;
        RMSNorm norm{nullptr};
        torch::nn::Linear output{nullptr};
        torch::Tensor freqs_cis;

        explicit TransformerImpl(const ModelArgs &params);

        torch::Tensor forward(
            const torch::Tensor &tokens,
            int64_t start_pos = 0,
            const std::vector<KVCache> &caches = {},
            const torch::optional<torch::Tensor> &targets = torch::nullopt
        );

        std::unique_ptr<torch::optim::AdamW> configure_optimizers(
            double weight_decay,
            double learning_rate,
            std::pair<double, double> betas,
            const std::string &device_type
        ) const;

        double estimate_mfu(int64_t fwdbwd_per_iter, double dt) const;

        torch::Tensor generate(
            const torch::Tensor &idx,
            int64_t max_new_tokens,
            double temperature = 1.0,
            std::optional<int64_t> top_k = std::nullopt
        );

    private:
        static void _init_weights(torch::nn::Module &module);
    };

    TORCH_MODULE (Transformer);
} // namespace Model

#endif // LEAP_TRANSFORMER_H