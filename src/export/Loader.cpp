#include "Loader.h"

#include <filesystem>
#include <fstream>
#include <iostream>
#include <vector>
#include <algorithm>
#include <map>
#include <regex>

// Third-party libraries
#include <nlohmann/json.hpp>
#include <safetensors.hh>

namespace fs = std::filesystem;
using json = nlohmann::json;

namespace Export {
    // Helper to map safetensors data types to torch::ScalarType
    static torch::ScalarType get_dtype(const safetensors::dtype type) {
        switch (type) {
            case safetensors::dtype::kFLOAT32: return torch::kFloat32;
            case safetensors::dtype::kFLOAT16: return torch::kFloat16;
            case safetensors::dtype::kBFLOAT16: return torch::kBFloat16;
            case safetensors::dtype::kINT64: return torch::kInt64;
            case safetensors::dtype::kINT32: return torch::kInt32;
            case safetensors::dtype::kINT8: return torch::kInt8;
            case safetensors::dtype::kUINT8: return torch::kUInt8;
            default: throw std::runtime_error("Unsupported safetensors dtype");
        }
    }

    // Loads a single .safetensors file into a map of Tensors.
    // Uses mmap internally via safetensors library.
    std::map<std::string, torch::Tensor> load_file(const std::string &path) {
        std::string warn, err;
        safetensors::safetensors_t st;

        if (!safetensors::load_from_file(path, &st, &warn, &err)) {
            throw std::runtime_error("Failed to load safetensors file " + path + ": " + err);
        }

        std::map<std::string, torch::Tensor> state_dict;
        uint8_t *base_ptr = st.storage.data();

        for (const std::vector<std::string> &keys = st.tensors.keys(); const auto &name: keys) {
            safetensors::tensor_t tensor_info;
            st.tensors.at(name, &tensor_info);

            std::vector<int64_t> sizes;
            sizes.reserve(tensor_info.shape.size());
            for (const auto d: tensor_info.shape) {
                sizes.push_back(static_cast<int64_t>(d));
            }

            auto options = torch::TensorOptions().dtype(get_dtype(tensor_info.dtype));
            const size_t offset = tensor_info.data_offsets[0];
            const auto data_ptr = static_cast<void *>(base_ptr + offset);

            // Create tensor from blob.
            // .clone() is ESSENTIAL here because safetensors_t destructor might unmap memory.
            // Ideally, we'd keep the file mapped, but for export simplicity we copy.
            auto t = torch::from_blob(data_ptr, sizes, options);
            state_dict[name] = t.clone();
        }

        return state_dict;
    }

    // Reverse permutation for RoPE weights (WQ, WK)
    // Optimized to reduce intermediate views
    torch::Tensor permute_reverse(const torch::Tensor &w, const int64_t n_heads, int64_t dim1, int64_t dim2) {
        // Python: w.view(n_heads, 2, dim1 // n_heads // 2, dim2).transpose(1, 2).reshape(dim1, dim2)
        // We can do this with efficient reshaping
        return w.view({n_heads, 2, dim1 / n_heads / 2, dim2})
                .transpose(1, 2)
                .reshape({dim1, dim2})
                .contiguous();
    }

    Model::Transformer load_meta_model(const std::string &model_path) {
        fs::path base_path(model_path);
        fs::path params_path = base_path / "params.json";

        std::ifstream f(params_path);
        if (!f.is_open()) {
            throw std::runtime_error("Could not open params.json at " + params_path.string());
        }
        json params;
        f >> params;
        std::cout << "Loaded params: " << params.dump() << std::endl;

        std::vector<fs::path> model_paths;
        for (const auto &entry: fs::directory_iterator(base_path)) {
            if (entry.path().extension() == ".safetensors") {
                model_paths.push_back(entry.path());
            }
        }
        if (model_paths.empty()) {
            throw std::runtime_error("No .safetensors files found in " + model_path);
        }
        std::ranges::sort(model_paths);

        std::cout << "Loading " << model_paths.size() << " safetensor files..." << std::endl;

        // Parallel load of shards
        std::vector<std::map<std::string, torch::Tensor> > shards(model_paths.size());
#pragma omp parallel for
        for (size_t i = 0; i < model_paths.size(); ++i) {
            shards[i] = load_file(model_paths[i].string());
        }

        std::map<std::string, torch::Tensor> state_dict;
        // Collect keys from first shard
        std::vector<std::string> keys;
        for (const auto &key: shards[0] | std::views::keys) {
            keys.push_back(key);
        }

        std::cout << "Concatenating weights..." << std::endl;

        // This loop can be slow if map insertions are frequent.
        // However, since we process key-by-key, we can't easily parallelize the outer loop 
        // without locking the state_dict.
        // Parallelizing tensor concatenation is handled by Torch internally.
        for (const auto &name: keys) {
            std::vector<torch::Tensor> tensors;
            tensors.reserve(shards.size());

            for (auto &shard: shards) {
                if (auto it = shard.find(name); it != shard.end()) {
                    tensors.push_back(std::move(it->second));
                    shard.erase(it); // release memory ASAP
                }
            }

            if (tensors.empty()) continue;

            if (tensors.size() == 1 || tensors[0].ndimension() == 1) {
                state_dict[name] = std::move(tensors[0]);
            } else {
                bool is_axis_1 = (
                    name.rfind("tok_embeddings.", 0) == 0 ||
                    (name.size() >= 20 && name.compare(name.size() - 20, 20, ".attention.wo.weight") == 0) ||
                    (name.size() >= 23 && name.compare(name.size() - 23, 23, ".feed_forward.w2.weight") == 0)
                );

                int64_t axis = is_axis_1 ? 1 : 0;
                state_dict[name] = torch::cat(tensors, axis);
            }
        }
        shards.clear(); // Free empty maps

        // Handle HF -> Meta keys
        bool has_tok = state_dict.contains("tok_embeddings.weight");

        if (bool has_embed = state_dict.contains("model.embed_tokens.weight"); !has_tok && has_embed) {
            std::cout << "Transforming Hugging Face keys to Meta format..." << std::endl;
            std::map<std::string, torch::Tensor> new_state_dict;

            int64_t n_heads = params["n_heads"].get<int64_t>();
            int64_t n_kv_heads = params.contains("n_kv_heads") ? params["n_kv_heads"].get<int64_t>() : n_heads;
            int64_t dim = params["dim"].get<int64_t>();

            // Direct moves where possible
            auto move_key = [&](const std::string &src, const std::string &dst) {
                if (state_dict.contains(src)) {
                    new_state_dict[dst] = std::move(state_dict[src]);
                }
            };

            move_key("model.embed_tokens.weight", "tok_embeddings.weight");
            move_key("model.norm.weight", "norm.weight");

            if (state_dict.contains("lm_head.weight")) {
                move_key("lm_head.weight", "output.weight");
            } else {
                // Weight tying: assuming clone is needed if it's shared? 
                // Actually in export we just need the data.
                new_state_dict["output.weight"] = new_state_dict["tok_embeddings.weight"];
            }

            // Layer Mappings
            // We iterate over the old map. Since we are moving out of it, we should be careful.
            // But here we construct new keys from old keys.
            // It's safer to iterate keys copy or just iterate safely.
            // Since we move `value`, the original `state_dict` becomes invalid/empty values.

            // To be safe and optimal: iterate a copy of keys or use the fact that we're building a new map.
            // The logic involves string parsing which is fast enough.

            // Collect keys first to avoid iterator invalidation or weirdness (though moving value is fine)
            // But we can just iterate.

            // Note: Parallelizing this loop requires locking new_state_dict, probably not worth it for metadata ops.
            for (auto &[key, value]: state_dict) {
                if (value.defined() == false) continue; // Already moved
                if (key.find("model.layers.") == std::string::npos) continue;

                // Parsing logic ...
                std::vector<std::string> parts;
                std::stringstream ss(key);
                std::string segment;
                while (std::getline(ss, segment, '.')) parts.push_back(segment);

                if (parts.size() < 4) continue;

                int layer_i = -1;
                try { layer_i = std::stoi(parts[2]); } catch (...) { continue; }

                std::string suffix;
                for (size_t k = 3; k < parts.size(); ++k) {
                    suffix += parts[k];
                    if (k < parts.size() - 1) suffix += ".";
                }

                std::string prefix = "layers." + std::to_string(layer_i) + ".";

                if (suffix == "input_layernorm.weight") {
                    new_state_dict[prefix + "attention_norm.weight"] = std::move(value);
                } else if (suffix == "post_attention_layernorm.weight") {
                    new_state_dict[prefix + "ffn_norm.weight"] = std::move(value);
                } else if (suffix == "self_attn.q_proj.weight") {
                    new_state_dict[prefix + "attention.wq.weight"] = permute_reverse(value, n_heads, dim, dim);
                } else if (suffix == "self_attn.k_proj.weight") {
                    int64_t kv_dim = (dim * n_kv_heads) / n_heads;
                    new_state_dict[prefix + "attention.wk.weight"] = permute_reverse(value, n_kv_heads, kv_dim, dim);
                } else if (suffix == "self_attn.v_proj.weight") {
                    new_state_dict[prefix + "attention.wv.weight"] = std::move(value);
                } else if (suffix == "self_attn.o_proj.weight") {
                    new_state_dict[prefix + "attention.wo.weight"] = std::move(value);
                } else if (suffix == "mlp.gate_proj.weight") {
                    new_state_dict[prefix + "feed_forward.w1.weight"] = std::move(value);
                } else if (suffix == "mlp.down_proj.weight") {
                    new_state_dict[prefix + "feed_forward.w2.weight"] = std::move(value);
                } else if (suffix == "mlp.up_proj.weight") {
                    new_state_dict[prefix + "feed_forward.w3.weight"] = std::move(value);
                }
            }
            state_dict = std::move(new_state_dict);
        }

        Model::ModelArgs config;
        config.dim = params["dim"].get<int64_t>();
        config.n_layers = params["n_layers"].get<int64_t>();
        config.n_heads = params["n_heads"].get<int64_t>();
        config.n_kv_heads = params.contains("n_kv_heads") ? params["n_kv_heads"].get<int64_t>() : config.n_heads;
        config.multiple_of = params["multiple_of"].get<int64_t>();
        config.norm_eps = params["norm_eps"].get<double>();
        config.vocab_size = state_dict["tok_embeddings.weight"].size(0);
        config.max_seq_len = 2048;

        std::cout << "Initializing Transformer model..." << std::endl;
        Model::Transformer model(config);

        torch::NoGradGuard no_grad;

        auto load_param = [&](const torch::Tensor &param, const std::string &key) {
            if (state_dict.contains(key)) {
                param.set_data(state_dict[key]);
            } else {
                std::cerr << "Warning: Missing key " << key << std::endl;
            }
        };

        load_param(model->tok_embeddings->weight, "tok_embeddings.weight");
        load_param(model->norm->weight, "norm.weight");
        load_param(model->output->weight, "output.weight");

        // Use parallel loading for layers?
        // set_data is fast (pointer assignment), but maybe iteration overhead.
        // Sequential is fine for safety.
        for (int i = 0; i < model->layers.size(); ++i) {
            auto &layer = model->layers[i];
            std::string prefix = "layers." + std::to_string(i) + ".";

            load_param(layer->attention_norm->weight, prefix + "attention_norm.weight");
            load_param(layer->attention->wq->weight, prefix + "attention.wq.weight");
            load_param(layer->attention->wk->weight, prefix + "attention.wk.weight");
            load_param(layer->attention->wv->weight, prefix + "attention.wv.weight");
            load_param(layer->attention->wo->weight, prefix + "attention.wo.weight");

            load_param(layer->ffn_norm->weight, prefix + "ffn_norm.weight");
            load_param(layer->feed_forward->w1->weight, prefix + "feed_forward.w1.weight");
            load_param(layer->feed_forward->w2->weight, prefix + "feed_forward.w2.weight");
            load_param(layer->feed_forward->w3->weight, prefix + "feed_forward.w3.weight");
        }

        model->eval();

        // Automatic Device Selection
        torch::Device device(torch::kCPU);
        if (torch::cuda::is_available()) {
            std::cout << "CUDA detected. Moving model to GPU..." << std::endl;
            device = torch::Device(torch::kCUDA);
        } else if (torch::mps::is_available()) {
            std::cout << "MPS detected. Moving model to Apple Silicon GPU..." << std::endl;
            device = torch::Device(torch::kMPS);
        }

        if (device.type() != torch::kCPU) {
            model->to(device);
        }

        std::cout << "Model loaded successfully." << std::endl;
        return model;
    }
} // namespace Model