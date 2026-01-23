#ifndef LEAP_CONFIG_H
#define LEAP_CONFIG_H

#include <string>

namespace Inference {
    struct Config {
        int dim; // transformer dimension
        int hidden_dim; // for ffn layers
        int n_layers; // number of layers
        int n_heads; // number of query heads
        int n_kv_heads; // number of key/value heads (can be < query heads because of multiquery)
        int vocab_size; // vocabulary size, usually 4096 (byte-level)
        int seq_len; // max sequence length
    };
} // namespace Inference

#endif //LEAP_CONFIG_H
