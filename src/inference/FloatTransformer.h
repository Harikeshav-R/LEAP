#ifndef LEAP_FLOAT_TRANSFORMER_H
#define LEAP_FLOAT_TRANSFORMER_H

#include "Config.h"
#include "Transformer.h"
#include <vector>

namespace Inference {
    struct FloatTransformerWeights {
        float *token_embedding_table;
        float *rms_att_weight;
        float *rms_ffn_weight;
        float *wq;
        float *wk;
        float *wv;
        float *wo;
        float *w1;
        float *w2;
        float *w3;
        float *rms_final_weight;
        float *wcls;
    };

    struct FloatRunState {
        std::vector<float> x;
        std::vector<float> xb;
        std::vector<float> xb2;
        std::vector<float> hb;
        std::vector<float> hb2;
        std::vector<float> q;
        std::vector<float> k;
        std::vector<float> v;
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

        void memory_map_weights(float *ptr, int shared_weights);

        void init_run_state();

        static void rmsnorm(float *o, const float *x, const float *weight, int size);

        static void softmax(float *x, int size);

        static void matmul(float *xout, const float *x, const float *w, int n, int d);
    };
} // namespace Inference

#endif // LEAP_FLOAT_TRANSFORMER_H