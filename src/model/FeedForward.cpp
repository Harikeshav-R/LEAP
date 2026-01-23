#include "FeedForward.h"

namespace Model {
    FeedForwardImpl::FeedForwardImpl(const int64_t dim, int64_t hidden_dim, const int64_t multiple_of, double dropout) {
        // Logic to handle "if hidden_dim is None"
        // In C++, we use a sentinel value (like -1 or 0) to detect this.
        if (hidden_dim <= 0) {
            hidden_dim = 4 * dim;
            hidden_dim = (2 * hidden_dim) / 3;
            // Round up to nearest multiple_of
            hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) / multiple_of);
        }

        // Initialize and register layers with bias=False
        w1 = register_module("w1", torch::nn::Linear(torch::nn::LinearOptions(dim, hidden_dim).bias(false)));
        w2 = register_module("w2", torch::nn::Linear(torch::nn::LinearOptions(hidden_dim, dim).bias(false)));
        w3 = register_module("w3", torch::nn::Linear(torch::nn::LinearOptions(dim, hidden_dim).bias(false)));

        dropout_layer = register_module("dropout", torch::nn::Dropout(dropout));
    }

    torch::Tensor FeedForwardImpl::forward(const torch::Tensor &x) {
        // Equivalent to: return self.dropout(self.w2(F.silu(self.w1(x)) * self.w3(x)))

        // 1. Calculate silu(w1(x))
        auto val = torch::nn::functional::silu(w1->forward(x));

        // 2. Multiply by w3(x)
        val = val * w3->forward(x);

        // 3. Project back via w2
        val = w2->forward(val);

        // 4. Apply dropout
        return dropout_layer->forward(val);
    }
} // namespace Model