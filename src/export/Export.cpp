#include "Export.h"

#include "Utils.h"
#include <iostream>
#include <fstream>
#include <vector>
#include <algorithm>
#include <future>

namespace Export {
    // Helper to write primitive types to file
    template<typename T>
    void write_raw(std::ofstream &out, T val) {
        out.write(reinterpret_cast<const char *>(&val), sizeof(T));
    }

    void float32_export(const Model::Transformer &model, const std::string &filepath) {
        constexpr int32_t version = 1;

        std::ofstream out_file(filepath, std::ios::binary);
        if (!out_file.is_open()) {
            std::cerr << "Failed to open file: " << filepath << std::endl;
            return;
        }

        // 1) Write magic (uint32 of "ak42")
        constexpr uint32_t magic = 0x616B3432;
        write_raw(out_file, magic);

        // 2) Write version
        write_raw(out_file, version);

        // 3) Write params
        const auto &p = model->params;
        // Access hidden_dim via the first layer's feed forward weight size
        const int32_t hidden_dim = model->layers[0]->feed_forward->w1->weight.size(0);
        const int32_t n_kv_heads = p.n_kv_heads.has_value() ? p.n_kv_heads.value() : p.n_heads;

        write_raw(out_file, static_cast<int32_t>(p.dim));
        write_raw(out_file, hidden_dim);
        write_raw(out_file, static_cast<int32_t>(p.n_layers));
        write_raw(out_file, static_cast<int32_t>(p.n_heads));
        write_raw(out_file, n_kv_heads);
        write_raw(out_file, static_cast<int32_t>(p.vocab_size));
        write_raw(out_file, static_cast<int32_t>(p.max_seq_len));

        // 4) Write flags
        const bool shared_classifier = torch::equal(model->tok_embeddings->weight, model->output->weight);
        write_raw(out_file, static_cast<uint8_t>(shared_classifier));

        // Pad with zeros to reach 256 bytes
        const long current_pos = out_file.tellp();
        if (const long pad = 256 - current_pos; pad > 0) {
            const std::vector<char> zeros(pad, 0);
            out_file.write(zeros.data(), pad);
        }

        // 5) Write weights
        // Order: Attention norms, FFN norms, Final norm, Token embeddings,
        //        Attn weights (wq, wk, wv, wo), FF weights (w1, w2, w3), Output (if not shared)

        // Attention Norms
        for (const auto &layer: model->layers) {
            serialize_fp32(out_file, layer->attention_norm->weight);
        }
        // FFN Norms
        for (const auto &layer: model->layers) {
            serialize_fp32(out_file, layer->ffn_norm->weight);
        }
        // Final Norm
        serialize_fp32(out_file, model->norm->weight);
        // Token Embeddings
        serialize_fp32(out_file, model->tok_embeddings->weight);

        // Attention Weights
        for (const auto &layer: model->layers) serialize_fp32(out_file, layer->attention->wq->weight);
        for (const auto &layer: model->layers) serialize_fp32(out_file, layer->attention->wk->weight);
        for (const auto &layer: model->layers) serialize_fp32(out_file, layer->attention->wv->weight);
        for (const auto &layer: model->layers) serialize_fp32(out_file, layer->attention->wo->weight);

        // Feed Forward Weights
        for (const auto &layer: model->layers) serialize_fp32(out_file, layer->feed_forward->w1->weight);
        for (const auto &layer: model->layers) serialize_fp32(out_file, layer->feed_forward->w2->weight);
        for (const auto &layer: model->layers) serialize_fp32(out_file, layer->feed_forward->w3->weight);

        // Output Weight
        if (!shared_classifier) {
            serialize_fp32(out_file, model->output->weight);
        }

        out_file.close();
        std::cout << "wrote " << filepath << std::endl;
    }

    void int8_export(const Model::Transformer &model, const std::string &filepath, int64_t group_size) {
        int32_t version = 2;
        const auto &p = model->params;

        // Validation: Backoff group size if necessary
        while (p.dim % group_size != 0) {
            group_size /= 2;
            std::cout << "BACKOFF: reducing group size to " << group_size << " to fit hidden_dim" << std::endl;
        }

        // Collect weights to be quantized
        std::vector<torch::Tensor> weights;
        weights.push_back(model->tok_embeddings->weight);
        for (const auto &layer: model->layers) weights.push_back(layer->attention->wq->weight);
        for (const auto &layer: model->layers) weights.push_back(layer->attention->wk->weight);
        for (const auto &layer: model->layers) weights.push_back(layer->attention->wv->weight);
        for (const auto &layer: model->layers) weights.push_back(layer->attention->wo->weight);
        for (const auto &layer: model->layers) weights.push_back(layer->feed_forward->w1->weight);
        for (const auto &layer: model->layers) weights.push_back(layer->feed_forward->w2->weight);
        for (const auto &layer: model->layers) weights.push_back(layer->feed_forward->w3->weight);

        bool shared_classifier = torch::equal(model->tok_embeddings->weight, model->output->weight);
        if (!shared_classifier) {
            weights.push_back(model->output->weight);
        }

        for (size_t i = 0; i < weights.size(); ++i) {
            if (weights[i].numel() % group_size != 0) {
                std::cerr << "Weight " << i << " has numel " << weights[i].numel()
                        << ", not a multiple of group_size " << group_size << std::endl;
                return; // or throw exception
            }
        }

        std::ofstream out_file(filepath, std::ios::binary);
        if (!out_file.is_open()) {
            std::cerr << "Failed to open file: " << filepath << std::endl;
            return;
        }

        // 1) Write Header
        uint32_t magic = 0x616B3432;
        write_raw(out_file, magic);
        write_raw(out_file, version);

        int32_t hidden_dim = model->layers[0]->feed_forward->w1->weight.size(0);
        int32_t n_kv_heads = p.n_kv_heads.has_value() ? p.n_kv_heads.value() : p.n_heads;

        write_raw(out_file, static_cast<int32_t>(p.dim));
        write_raw(out_file, hidden_dim);
        write_raw(out_file, static_cast<int32_t>(p.n_layers));
        write_raw(out_file, static_cast<int32_t>(p.n_heads));
        write_raw(out_file, n_kv_heads);
        write_raw(out_file, static_cast<int32_t>(p.vocab_size));
        write_raw(out_file, static_cast<int32_t>(p.max_seq_len));

        // Flags and group size
        write_raw(out_file, static_cast<uint8_t>(shared_classifier));
        write_raw(out_file, static_cast<int32_t>(group_size));

        // Padding
        long current_pos = out_file.tellp();
        long pad = 256 - current_pos;
        if (pad > 0) {
            std::vector<char> zeros(pad, 0);
            out_file.write(zeros.data(), pad);
        }

        // 2) Write FP32 Params (Norms)
        for (const auto &layer: model->layers) serialize_fp32(out_file, layer->attention_norm->weight);
        for (const auto &layer: model->layers) serialize_fp32(out_file, layer->ffn_norm->weight);
        serialize_fp32(out_file, model->norm->weight);

        // 3) Quantize and Write Weights
        std::vector<float> errors(weights.size());

        struct QuantResult {
            torch::Tensor q;
            torch::Tensor s;
            float err;
        };

        // Pipeline: 
        // We process weights one by one (or in small batches) to save memory, 
        // but we want to parallelize quantization of tensor N with writing of tensor N-1.

        // Since we already have all weights loaded in memory (from Loader.cpp), 
        // we can stick to the existing structure but optimize the write phase.
        // Actually, the previous code quantized ALL weights in parallel first, storing ALL results in RAM.
        // That is fast for compute but OOM-risky for 70B models (needs 2x model size in RAM).

        // Better approach for large models:
        // Stream processing: Quantize W[i] -> Write W[i] -> Free W[i]
        // Parallel approach: Background Writer.

        // However, keeping the current architecture (RAM rich assumption based on loading full model first):
        // The previous code:
        // 1. Quantize ALL (Parallel) -> high peak RAM, fast compute.
        // 2. Write ALL (Sequential) -> bottleneck.

        // We will keep the "Quantize ALL" approach if RAM permits (as implied by current Loader),
        // but since we are optimizing, let's assume we want to support huge models better 
        // OR just optimize the write speed.

        // Actually, the previous code quantized *everything* into `results` vector first.
        // Then wrote everything. 
        // This effectively requires 2x memory (FP32 weights + Int8 weights + Scales).

        // OPTIMIZED PIPELINE (Low Memory + High Speed):
        // Loop i from 0 to N:
        //   Launch Async Quantize(i+1)
        //   Write(i) (if i >= 0)
        //   Join Quantize(i+1)

        // Let's implement a simple Lookahead Pipeline.

        std::cout << "Quantizing and writing weights (Pipelined)..." << std::endl;

        // We will use a double-buffer slot approach.
        // Current ready result
        QuantResult current_res;
        bool current_ready = false;

        // Future for next result
        std::future<QuantResult> next_res_future;

        auto launch_quantize = [&](size_t idx) {
            return std::async(std::launch::async, [=, &weights]() {
                auto result = quantize_q80(weights[idx], group_size);
                return QuantResult{std::get < 0 > (result), std::get < 1 > (result), std::get < 2 > (result)};
            });
        };

        // Start first job
        if (!weights.empty()) {
            next_res_future = launch_quantize(0);
        }

        for (size_t i = 0; i < weights.size(); ++i) {
            // Wait for current quantization to finish
            if (next_res_future.valid()) {
                current_res = next_res_future.get();
                current_ready = true;
            }

            // Prefetch/Start next quantization immediately
            if (i + 1 < weights.size()) {
                next_res_future = launch_quantize(i + 1);
            }

            // Now Write Current (IO Bound)
            // While this writes, the next quantization (CPU Bound) is happening in background thread
            const auto &w = weights[i]; // Original weight for logging shape

            serialize_int8(out_file, current_res.q);
            serialize_fp32(out_file, current_res.s);

            errors[i] = current_res.err;

            // Logging
            std::string shape_str = "(";
            for (int k = 0; k < w.dim(); ++k) shape_str += std::to_string(w.size(k)) + (k < w.dim() - 1 ? ", " : "");
            shape_str += ")";

            std::cout << (i + 1) << "/" << weights.size() << " saved " << shape_str
                    << " (max error " << current_res.err << ")" << std::endl;

            // Optional: Free the original weight memory if we own it uniquely to save RAM?
            // Since `weights` vector holds copies or views, usually we can't easily free here 
            // without destructing the tensor, but `weights[i] = torch::Tensor();` might help if refs allow.
            // keeping it simple.
        }

        if (!errors.empty()) {
            std::ranges::sort(errors, std::greater<float>());
            std::cout << "max quantization group error across all weights: " << errors[0] << std::endl;
        }

        out_file.close();
        std::cout << "wrote " << filepath << std::endl;
    }
}