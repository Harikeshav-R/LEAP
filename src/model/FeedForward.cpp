#include "FeedForward.h"

namespace Model {
    FeedForwardImpl::FeedForwardImpl(const int64_t dim, int64_t hidden_dim, const int64_t multiple_of, double dropout) {
        if (hidden_dim <= 0) {
            hidden_dim = 4 * dim;
            hidden_dim = (2 * hidden_dim) / 3;
            hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) / multiple_of);
        }

        w1 = register_module("w1", torch::nn::Linear(torch::nn::LinearOptions(dim, hidden_dim).bias(false)));
        w2 = register_module("w2", torch::nn::Linear(torch::nn::LinearOptions(hidden_dim, dim).bias(false)));
        w3 = register_module("w3", torch::nn::Linear(torch::nn::LinearOptions(dim, hidden_dim).bias(false)));

        dropout_layer = register_module("dropout", torch::nn::Dropout(dropout));
    }

    torch::Tensor FeedForwardImpl::forward(const torch::Tensor &x) {
        // SwiGLU: w2(silu(w1(x)) * w3(x))
        
        // 1. Calculate w1(x)
        auto val = w1->forward(x);
        
        // 2. Apply SiLU in-place
        torch::silu_(val);
        
        // 3. Multiply by w3(x) in-place
        val.mul_(w3->forward(x));
        
        // 4. Project back via w2
        val = w2->forward(val);
        
        // 5. Dropout
        return dropout_layer->forward(val);
    }
} // namespace Model
