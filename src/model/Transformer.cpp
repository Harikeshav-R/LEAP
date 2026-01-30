#include "Transformer.h"
#include <cmath>
#include "Utils.h"

namespace Model {
    TransformerImpl::TransformerImpl(const ModelArgs &params) : params(params) {
        tok_embeddings = register_module("tok_embeddings",
                                         torch::nn::Embedding(
                                             torch::nn::EmbeddingOptions(params.vocab_size, params.dim)));

        dropout = register_module("dropout", torch::nn::Dropout(params.dropout));

        for (int64_t i = 0; i < params.n_layers; ++i) {
            auto block = TransformerBlock(i, params);
            layers.push_back(block);
            register_module("layers_" + std::to_string(i), block);
        }

        norm = register_module("norm", RMSNorm(params.dim, params.norm_eps));

        output = register_module("output",
                                 torch::nn::Linear(
                                     torch::nn::LinearOptions(params.dim, params.vocab_size).bias(false)));

        tok_embeddings->weight = output->weight;

        const auto freqs = precompute_freqs_cis(
            params.dim / params.n_heads,
            params.max_seq_len
        );

        freqs_cis = register_buffer("freqs_cis", freqs);

        this->apply([this](torch::nn::Module &module) {
            this->_init_weights(module);
        });

        for (auto named_params = this->named_parameters(true); auto &k_v: named_params) {
            const std::string &pn = k_v.key();
            const torch::Tensor &p = k_v.value();
            if (pn.find("w3.weight") != std::string::npos || pn.find("wo.weight") != std::string::npos) {
                torch::nn::init::normal_(p, 0.0, 0.02 / std::sqrt(2 * params.n_layers));
            }
        }
    }

    void TransformerImpl::_init_weights(torch::nn::Module &module) {
        if (module.as<torch::nn::Linear>()) {
            const auto linear = module.as<torch::nn::Linear>();
            torch::nn::init::normal_(linear->weight, 0.0, 0.02);
            if (linear->bias.defined()) {
                torch::nn::init::zeros_(linear->bias);
            }
        } else if (module.as<torch::nn::Embedding>()) {
            const auto embedding = module.as<torch::nn::Embedding>();
            torch::nn::init::normal_(embedding->weight, 0.0, 0.02);
        }
    }

    torch::Tensor TransformerImpl::forward(
        const torch::Tensor &tokens,
        int64_t start_pos,
        const std::vector<KVCache> &caches,
        const torch::optional<torch::Tensor> &targets
    ) {
        const auto seqlen = tokens.size(1);
        auto h = dropout->forward(tok_embeddings->forward(tokens));

        // Slice frequencies: freqs_cis[start_pos : start_pos + seqlen]
        const auto current_freqs_cis = freqs_cis.slice(0, start_pos, start_pos + seqlen);

        for (size_t i = 0; i < layers.size(); ++i) {
            std::optional<KVCache> layer_cache;
            if (i < caches.size()) {
                layer_cache = caches[i];
            }
            h = layers[i]->forward(h, current_freqs_cis, start_pos, layer_cache);
        }
        h = norm->forward(h);

        torch::Tensor logits;
        if (targets.has_value()) {
            logits = output->forward(h);
            const auto logits_view = logits.view({-1, logits.size(-1)});
            const auto targets_view = targets.value().view({-1});
            this->last_loss = torch::nn::functional::cross_entropy(
                logits_view,
                targets_view,
                torch::nn::functional::CrossEntropyFuncOptions().ignore_index(-1)
            );
        } else {
            logits = output->forward(h.slice(1, -1));
            this->last_loss = torch::nullopt;
        }

        return logits;
    }

    torch::Tensor TransformerImpl::generate(
        const torch::Tensor &idx,
        const int64_t max_new_tokens,
        const double temperature,
        const std::optional<int64_t> top_k
    ) {
        torch::InferenceMode guard(true);

        const int64_t B = idx.size(0);
        const int64_t T = idx.size(1);
        const int64_t T_new = T + max_new_tokens;
        const int64_t max_seq_len = params.max_seq_len;
        const int64_t n_kv_heads = params.n_kv_heads.value_or(params.n_heads);
        const int64_t head_dim = params.dim / params.n_heads;

        // Initialize KV Cache
        // Each layer gets a cache of shape [B, n_kv_heads, max_seq_len, head_dim]
        // Pre-allocated on the correct device/dtype
        std::vector<KVCache> caches;
        caches.reserve(params.n_layers);
        for (int i = 0; i < params.n_layers; ++i) {
            caches.push_back({
                torch::zeros({B, n_kv_heads, max_seq_len, head_dim}, idx.options().dtype(torch::kFloat)),
                torch::zeros({B, n_kv_heads, max_seq_len, head_dim}, idx.options().dtype(torch::kFloat))
            });
        }

        auto idx_buffer = torch::empty({B, T_new}, idx.options());
        idx_buffer.slice(1, 0, T).copy_(idx);

        // Current input tensor. For step 0, it's the full prompt.
        // For step > 0, it will be the single token generated in the previous step.
        torch::Tensor idx_cond = idx;

        for (int64_t i = 0; i < max_new_tokens; ++i) {
            const int64_t start_pos = (i == 0) ? 0 : (T + i - 1);
            int64_t end_pos = T + i; // Used only for buffer writing

            // Forward pass with cache
            torch::Tensor logits = this->forward(idx_cond, start_pos, caches);
            logits = logits.select(1, -1);

            torch::Tensor idx_next;

            if (temperature == 0.0) {
                auto max_result = logits.topk(1, -1);
                idx_next = std::get < 1 > (max_result);
            } else {
                if (std::abs(temperature - 1.0) > 1e-6) {
                    logits /= temperature;
                }
                if (top_k.has_value()) {
                    const int64_t k = std::min(top_k.value(), logits.size(-1));
                    auto topk_result = logits.topk(k);
                    auto v = std::get < 0 > (topk_result);
                    auto pivot = v.select(1, -1).unsqueeze(1);
                    logits = torch::where(logits < pivot,
                                          torch::tensor(-std::numeric_limits<float>::infinity(), logits.options()),
                                          logits);
                }
                auto probs = torch::nn::functional::softmax(logits, -1);
                idx_next = torch::multinomial(probs, 1);
            }

            // Write to buffer
            idx_buffer.slice(1, end_pos, end_pos + 1).copy_(idx_next);

            // For the next iteration, input is just the generated token
            idx_cond = idx_next;
        }

        return idx_buffer;
    }

    // Copied from previous turn (no changes needed, just keeping file complete)
    std::unique_ptr<torch::optim::AdamW> TransformerImpl::configure_optimizers(
        double weight_decay,
        double learning_rate,
        std::pair<double, double> betas,
        const std::string &device_type
    ) const {
        std::vector<torch::Tensor> decay_params;
        std::vector<torch::Tensor> nodecay_params;
        for (auto named_params = this->named_parameters(true); auto &k_v: named_params) {
            auto &p = k_v.value();
            if (!p.requires_grad()) continue;
            if (p.dim() >= 2) decay_params.push_back(p);
            else nodecay_params.push_back(p);
        }
        std::vector<torch::optim::OptimizerParamGroup> groups;
        auto decay_opts = std::make_unique<torch::optim::AdamWOptions>(learning_rate);
        decay_opts->weight_decay(weight_decay).betas(betas);
        groups.emplace_back(decay_params, std::move(decay_opts));
        auto nodecay_opts = std::make_unique<torch::optim::AdamWOptions>(learning_rate);
        nodecay_opts->weight_decay(0.0).betas(betas);
        groups.emplace_back(nodecay_params, std::move(nodecay_opts));
        torch::optim::AdamWOptions defaults(learning_rate);
        defaults.betas(betas).weight_decay(weight_decay);
        return std::make_unique<torch::optim::AdamW>(groups, defaults);
    }

    double TransformerImpl::estimate_mfu(const int64_t fwdbwd_per_iter, const double dt) const {
        int64_t N = 0;
        for (const auto &p: this->parameters()) N += p.numel();
        auto &cfg = this->params;
        const double flops = 6.0 * N + 12.0 * cfg.n_layers * cfg.n_heads * (cfg.dim / cfg.n_heads) * cfg.max_seq_len;
        return (flops * cfg.max_seq_len * fwdbwd_per_iter * (1.0 / dt)) / 312e12;
    }
} // namespace Model