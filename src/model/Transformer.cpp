#include "Transformer.h"

#include <cmath>
#include <iostream>

#include "Utils.h"

namespace Model {
    TransformerImpl::TransformerImpl(const ModelArgs &params) : params(params) {
        // Initialize Submodules
        tok_embeddings = register_module("tok_embeddings",
                                         torch::nn::Embedding(
                                             torch::nn::EmbeddingOptions(params.vocab_size, params.dim)));

        dropout = register_module("dropout",
                                  torch::nn::Dropout(params.dropout));

        // Initialize layers
        for (int64_t i = 0; i < params.n_layers; ++i) {
            auto block = TransformerBlock(i, params);
            layers.push_back(block);
            register_module("layers_" + std::to_string(i), block);
        }

        norm = register_module("norm", RMSNorm(params.dim, params.norm_eps));

        output = register_module("output",
                                 torch::nn::Linear(
                                     torch::nn::LinearOptions(params.dim, params.vocab_size).bias(false)));

        // Weight Tying: share the unembedding parameters with the embedding parameters
        // In LibTorch, we simply assign the tensor handle.
        tok_embeddings->weight = output->weight;

        // Precompute RoPE frequencies
        auto [fst, snd] = precompute_freqs_cis(
            params.dim / params.n_heads,
            params.max_seq_len
        );

        // Register buffers (persistent=False equivalent is not saving them in state_dict,
        // but typically register_buffer saves them. Here we just register them so they move with device).
        freqs_cos = register_buffer("freqs_cos", fst);
        freqs_sin = register_buffer("freqs_sin", snd);

        // Initialize all weights
        // We use a lambda to bind 'this' to access _init_weights
        this->apply([this](torch::nn::Module &module) {
            this->_init_weights(module);
        });

        // Apply special scaled init to the residual projections, per GPT-2 paper
        // We iterate over named parameters to find specific ones
        for (auto named_params = this->named_parameters(true /* recurse */); auto &k_v: named_params) {
            const std::string &pn = k_v.key();
            const torch::Tensor &p = k_v.value();

            if (pn.find("w3.weight") != std::string::npos || pn.find("wo.weight") != std::string::npos) {
                torch::nn::init::normal_(
                    p, 0.0, 0.02 / std::sqrt(2 * params.n_layers)
                );
            }
        }

        last_loss = torch::nullopt;
    }

    
} // namespace Model
