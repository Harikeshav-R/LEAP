#ifndef LEAP_FLOAT_TRANSFORMER_H
#define LEAP_FLOAT_TRANSFORMER_H

#include "Config.h"
#include "Transformer.h"
#include <vector>

namespace Inference {
    struct FloatTransformerWeights {
        const float *token_embedding_table;
        const float *rms_att_weight;
        const float *rms_ffn_weight;
        const float *wq;
        const float *wk;
        const float *wv;
        const float *wo;
        const float *w1;
        const float *w2;
        const float *w3;
        const float *rms_final_weight;
        const float *wcls;
    };

    struct FloatRunState {
        std::vector<float> x;
        std::vector<float> xb;
        std::vector<float> xb2;
        std::vector<float> hb;
        std::vector<float> hb2;
        std::vector<float> q;
        // k and v are written directly to cache, no intermediate buffer needed
        std::vector<float> att;
        std::vector<float> logits;
        std::vector<float> key_cache;
        std::vector<float> value_cache;
    };

    class FloatTransformer : public Transformer {
    public:
        FloatTransformer(const Config &config, int fd, void *mmap_ptr, size_t file_size, float *weights_ptr,
                         int shared_weights);

        ~FloatTransformer() override;

        float *forward(int token, int pos) override;

    private:
        FloatTransformerWeights weights{};
        FloatRunState state{};
        int fd;
        void *mmap_ptr;
        size_t file_size;

        // RoPE lookup tables
        std::vector<float> rope_cos;
        std::vector<float> rope_sin;

        void memory_map_weights(float *ptr, int shared_weights);

        void init_run_state();

        void precompute_freqs();

        static void rmsnorm(float *o, const float *x, const float *weight, int size);

        static void softmax(float *x, int size);

        static void matmul(float *xout, const float *x, const float *w, int n, int d);
    };
} // namespace Inference

#endif // LEAP_FLOAT_TRANSFORMER_H