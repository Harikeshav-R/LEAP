#ifndef LEAP_MODELARGS_H
#define LEAP_MODELARGS_H

#include <optional>

namespace Model {
    struct ModelArgs {
        int dim = 4096;
        int n_layers = 32;
        int n_heads = 32;
        std::optional<int> n_kv_heads = std::nullopt;
        int vocab_size = 32000;
        std::optional<int> hidden_dim = std::nullopt;
        int multiple_of = 256;
        float norm_eps = 1e-5;
        int max_seq_len = 2048;
        float dropout = 0.0;
    };
}


#endif //LEAP_MODELARGS_H
