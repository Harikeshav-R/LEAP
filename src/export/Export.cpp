#include "Export.h"

#include "Utils.h"
#include <iostream>
#include <fstream>
#include <vector>
#include <algorithm>

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
        std::vector<QuantResult> results(weights.size());

        std::cout << "Quantizing " << weights.size() << " tensors (Parallel)..." << std::endl;
#pragma omp parallel for schedule(dynamic)
        for (size_t i = 0; i < weights.size(); ++i) {
            auto result = quantize_q80(weights[i], group_size);
            results[i] = {std::get<0>(result), std::get<1>(result), std::get<2>(result)};
        }

        std::cout << "Writing quantized weights..." << std::endl;
        for (size_t i = 0; i < weights.size(); ++i) {
            const auto &w = weights[i];
            const auto &res = results[i];

            serialize_int8(out_file, res.q);
            serialize_fp32(out_file, res.s);

            errors[i] = res.err;

            // Format shape for logging
            std::string shape_str = "(";
            for (int k = 0; k < w.dim(); ++k) shape_str += std::to_string(w.size(k)) + (k < w.dim() - 1 ? ", " : "");
            shape_str += ")";

            std::cout << (i + 1) << "/" << weights.size() << " saved " << shape_str
                    << " (max error " << res.err << ")" << std::endl;
        }

        if (!errors.empty()) {
            std::ranges::sort(errors, std::greater<float>());
            std::cout << "max quantization group error across all weights: " << errors[0] << std::endl;
        }

        out_file.close();
        std::cout << "wrote " << filepath << std::endl;
    }
}
