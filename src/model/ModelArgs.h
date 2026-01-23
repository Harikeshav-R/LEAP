#ifndef LEAP_MODELARGS_H
#define LEAP_MODELARGS_H

#include <optional>

namespace Model {
    struct ModelArgs {
        int64_t dim = 4096;
        int64_t n_layers = 32;
        int64_t n_heads = 32;
        std::optional<int64_t> n_kv_heads = std::nullopt;
        int64_t vocab_size = 32000;
        std::optional<int64_t> hidden_dim = std::nullopt;
        int64_t multiple_of = 256;
        double norm_eps = 1e-5;
        int64_t max_seq_len = 2048;
        double dropout = 0.0;
    };
}


#endif //LEAP_MODELARGS_H