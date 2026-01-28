#ifndef LEAP_FEEDFORWARD_H
#define LEAP_FEEDFORWARD_H

#include <torch/torch.h>

namespace Model {
    // Implementation class inheriting from torch::nn::Module
    struct FeedForwardImpl : torch::nn::Module {
        // Constructor
        // hidden_dim defaults to -1 to simulate python's 'None'
        FeedForwardImpl(int64_t dim, int64_t hidden_dim, int64_t multiple_of, double dropout);

        // Forward pass
        torch::Tensor forward(const torch::Tensor &x);

        // Submodules
        torch::nn::Linear w1{nullptr};
        torch::nn::Linear w2{nullptr};
        torch::nn::Linear w3{nullptr};
        torch::nn::Dropout dropout_layer{nullptr};
    };

    // TORCH_MODULE generates the 'FeedForward' wrapper class for reference semantics
    TORCH_MODULE (FeedForward);
} // namespace Model

#endif //LEAP_FEEDFORWARD_H