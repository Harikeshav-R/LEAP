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
    // =========================================================================
    // Helpers: Safetensors Loading
    // =========================================================================

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
    // NOTE: This implementation reads the file into memory to ensure data persists
    // after the function returns (via .clone()).
    // Loads a single .safetensors file into a map of Tensors.
    // Loads a single .safetensors file into a map of Tensors.
    std::map<std::string, torch::Tensor> load_file(const std::string &path) {
        std::string warn, err;
        safetensors::safetensors_t st;

        // Load the parsed safetensors structure
        if (!safetensors::load_from_file(path, &st, &warn, &err)) {
            throw std::runtime_error("Failed to load safetensors file " + path + ": " + err);
        }

        std::map<std::string, torch::Tensor> state_dict;
        uint8_t *base_ptr = st.storage.data();

        // FIX: Iterate using the keys() method provided by the ordered_dict
        // and look up the tensor info for each key.

        for (const std::vector<std::string> &keys = st.tensors.keys(); const auto &name: keys) {
            // Retrieve the tensor info for this key
            // Note: We access the underlying map/lookup via the ordered_dict's mechanism
            // If direct access isn't exposed, we iterate the 'list' member (if available),
            // but since you see keys(), there is likely an 'at()' or '[]' operator.
            // However, the safest access in this specific library structure (minijson::ordered_dict)
            // is often just iterating the internal vector if 'keys()' is separate.

            // To be 100% safe with this specific library variation without guessing the 'at' syntax:
            // We will iterate the underlying list which correlates to these keys.
            // But if you strictly want to use the keys found by clang:

            safetensors::tensor_t tensor_info;
            st.tensors.at(name, &tensor_info); // Assuming standard map-like access

            // Convert shapes
            std::vector<int64_t> sizes;
            for (const auto d: tensor_info.shape) {
                sizes.push_back(static_cast<int64_t>(d));
            }

            auto options = torch::TensorOptions().dtype(get_dtype(tensor_info.dtype));
            const size_t offset = tensor_info.data_offsets[0];
            auto *data_ptr = static_cast<void *>(base_ptr + offset);

            const torch::Tensor t = torch::from_blob(data_ptr, sizes, options).clone();
            state_dict[name] = t;
        }

        // Iterate directly over the ordered_dict
        // The iterator provides a pair/struct with 'first' (key) and 'second' (value)

        return state_dict;
    }

    // =========================================================================
    // Helpers: Weight Manipulation
    // =========================================================================

    // Reverse permutation for RoPE weights (WQ, WK)
    // Python: w.view(n_heads, 2, dim1 // n_heads // 2, dim2).transpose(1, 2).reshape(dim1, dim2)
    torch::Tensor permute_reverse(const torch::Tensor &w, const int64_t n_heads, int64_t dim1, int64_t dim2) {
        const auto view_shape = std::vector<int64_t>{n_heads, 2, dim1 / n_heads / 2, dim2};
        return w.view(view_shape)
                .transpose(1, 2)
                .reshape({dim1, dim2})
                .contiguous();
    }

    // =========================================================================
    // Main Loader Logic
    // =========================================================================

    Model::Transformer load_meta_model(const std::string &model_path) {
        fs::path base_path(model_path);

        // 1. Load params.json
        fs::path params_path = base_path / "params.json";
        std::ifstream f(params_path);
        if (!f.is_open()) {
            throw std::runtime_error("Could not open params.json at " + params_path.string());
        }
        json params;
        f >> params;
        std::cout << "Loaded params: " << params.dump() << std::endl;

        // 2. Find .safetensors files
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

        // 3. Load all shards
        std::cout << "Loading " << model_paths.size() << " safetensor files..." << std::endl;
        std::vector<std::map<std::string, torch::Tensor>> shards(model_paths.size());

        #pragma omp parallel for
        for (size_t i = 0; i < model_paths.size(); ++i) {
            shards[i] = load_file(model_paths[i].string());
        }

        // 4. Concat Weights
        // We accumulate the final state_dict here
        std::map<std::string, torch::Tensor> state_dict;

        // Iterate over keys from the first shard
        // Note: We copy the keys list to avoid iterator invalidation issues if we were modifying shards in place
        std::vector<std::string> keys;
        for (const auto &key: shards[0] | std::views::keys) {
            keys.push_back(key);
        }

        std::cout << "Concatenating weights..." << std::endl;
        for (const auto &name: keys) {
            std::vector<torch::Tensor> tensors;
            tensors.reserve(shards.size());

            // Collect this tensor from all shards
            for (auto &shard: shards) {
                if (shard.contains(name)) {
                    tensors.push_back(shard[name]);
                    shard.erase(name); // <--- CRITICAL: Drop ref count to free memory
                }
            }

            if (tensors.empty()) continue;

            // Determine concatenation axis
            if (tensors.size() == 1 || tensors[0].ndimension() == 1) {
                state_dict[name] = tensors[0];
            } else {
                bool is_axis_1 = (
                    name.rfind("tok_embeddings.", 0) == 0 || // startswith
                    (name.size() >= 20 && name.compare(name.size() - 20, 20, ".attention.wo.weight") == 0) ||
                    // endswith
                    (name.size() >= 23 && name.compare(name.size() - 23, 23, ".feed_forward.w2.weight") == 0)
                    // endswith
                );

                int64_t axis = is_axis_1 ? 1 : 0;
                state_dict[name] = torch::cat(tensors, axis);
            }
        }

        // Clear shards to free memory
        shards.clear();

        // 5. Handle Hugging Face keys -> Meta format conversion
        bool has_tok = state_dict.contains("tok_embeddings.weight");

        if (bool has_embed = state_dict.contains("model.embed_tokens.weight"); !has_tok && has_embed) {
            std::cout << "Transforming Hugging Face keys to Meta format..." << std::endl;
            std::map<std::string, torch::Tensor> new_state_dict;

            int64_t n_heads = params["n_heads"].get<int64_t>();
            int64_t n_kv_heads = params.contains("n_kv_heads") ? params["n_kv_heads"].get<int64_t>() : n_heads;
            int64_t dim = params["dim"].get<int64_t>();

            // Direct Mappings
            new_state_dict["tok_embeddings.weight"] = state_dict["model.embed_tokens.weight"];
            new_state_dict["norm.weight"] = state_dict["model.norm.weight"];

            if (state_dict.contains("lm_head.weight")) {
                new_state_dict["output.weight"] = state_dict["lm_head.weight"];
            } else {
                std::cout << "lm_head.weight not found, assuming tied weights..." << std::endl;
                new_state_dict["output.weight"] = state_dict["model.embed_tokens.weight"];
            }

            // Layer Mappings
            for (const auto &[key, value]: state_dict) {
                if (key.find("model.layers.") == std::string::npos) continue;

                // Split key: model.layers.{i}.{suffix}
                std::vector<std::string> parts;
                std::stringstream ss(key);
                std::string segment;
                while (std::getline(ss, segment, '.')) parts.push_back(segment);

                // Expect at least model, layers, i, ...
                if (parts.size() < 4) continue;

                // Extract layer index
                int layer_i = -1;
                try {
                    layer_i = std::stoi(parts[2]);
                } catch (...) { continue; }

                // Reconstruct suffix
                std::string suffix = "";
                for (size_t k = 3; k < parts.size(); ++k) {
                    suffix += parts[k];
                    if (k < parts.size() - 1) suffix += ".";
                }

                std::string prefix = "layers." + std::to_string(layer_i) + ".";

                if (suffix == "input_layernorm.weight") {
                    new_state_dict[prefix + "attention_norm.weight"] = value;
                } else if (suffix == "post_attention_layernorm.weight") {
                    new_state_dict[prefix + "ffn_norm.weight"] = value;
                } else if (suffix == "self_attn.q_proj.weight") {
                    new_state_dict[prefix + "attention.wq.weight"] = permute_reverse(value, n_heads, dim, dim);
                } else if (suffix == "self_attn.k_proj.weight") {
                    int64_t kv_dim = (dim * n_kv_heads) / n_heads;
                    new_state_dict[prefix + "attention.wk.weight"] = permute_reverse(value, n_kv_heads, kv_dim, dim);
                } else if (suffix == "self_attn.v_proj.weight") {
                    new_state_dict[prefix + "attention.wv.weight"] = value;
                } else if (suffix == "self_attn.o_proj.weight") {
                    new_state_dict[prefix + "attention.wo.weight"] = value;
                } else if (suffix == "mlp.gate_proj.weight") {
                    new_state_dict[prefix + "feed_forward.w1.weight"] = value;
                } else if (suffix == "mlp.down_proj.weight") {
                    new_state_dict[prefix + "feed_forward.w2.weight"] = value;
                } else if (suffix == "mlp.up_proj.weight") {
                    new_state_dict[prefix + "feed_forward.w3.weight"] = value;
                }
            }
            // Swap to new dictionary
            state_dict = new_state_dict;
        }

        // 6. Setup ModelArgs
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

        // 7. Load Weights into Model (In-place copy)
        // We use torch::NoGradGuard to prevent tracking this as an operation
        torch::NoGradGuard no_grad;

        // Helper to safely load a parameter
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

        for (int i = 0; i < model->layers.size(); ++i) {
            auto &layer = model->layers[i]; // Access the underlying TransformerBlock value
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
        std::cout << "Model loaded successfully." << std::endl;
        return model;
    }
} // namespace Model
