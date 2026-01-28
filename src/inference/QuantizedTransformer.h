#ifndef LEAP_QUANTIZED_TRANSFORMER_H
#define LEAP_QUANTIZED_TRANSFORMER_H

#include "Config.h"
#include "Transformer.h"
#include <vector>

namespace Inference {
    struct QuantizedTensor {
        int8_t *q; // quantized values
        float *s; // scaling factors
    };

    struct QuantizedTransformerWeights {
        std::vector<QuantizedTensor> q_tokens; // (vocab_size, dim)
        std::vector<float> token_embedding_table; // same, but dequantized

        float *rms_att_weight;
        float *rms_ffn_weight;

        std::vector<QuantizedTensor> wq;
        std::vector<QuantizedTensor> wk;
        std::vector<QuantizedTensor> wv;
        std::vector<QuantizedTensor> wo;

        std::vector<QuantizedTensor> w1;
        std::vector<QuantizedTensor> w2;
        std::vector<QuantizedTensor> w3;

        float *rms_final_weight;

        // If shared, points to q_tokens.data(). If not, points to wcls_storage.data().
        const QuantizedTensor *wcls;
        // Storage for non-shared classifier weights
        std::vector<QuantizedTensor> wcls_storage;
    };

    struct QuantizedRunState {
        std::vector<float> x;
        std::vector<float> xb;
        std::vector<float> xb2;
        std::vector<float> hb;
        std::vector<float> hb2;

        // Quantized buffers
        std::vector<int8_t> xq_q;
        std::vector<float> xq_s;

        std::vector<int8_t> hq_q;
        std::vector<float> hq_s;

        std::vector<float> q;
        std::vector<float> k; // Temporary buffer for current token Key
        std::vector<float> v; // Temporary buffer for current token Value
        std::vector<float> att;
        std::vector<float> logits;
        std::vector<float> key_cache;
        std::vector<float> value_cache;
    };

    class QuantizedTransformer : public Transformer {
    public:
        QuantizedTransformer(const Config &config, int fd, void *mmap_ptr, size_t file_size, void *weights_ptr,
                             int shared_weights, int group_size);

        ~QuantizedTransformer() override;

        float *forward(int token, int pos) override;

    private:
        QuantizedTransformerWeights weights{};
        QuantizedRunState state{};
        int fd;
        void *mmap_ptr;
        size_t file_size;
        int group_size; // GS

        // RoPE lookup tables
        std::vector<float> rope_cos;
        std::vector<float> rope_sin;

        void memory_map_weights(void *ptr, int shared_weights);

        void init_run_state();

        void precompute_freqs();

        // Helpers
        std::vector<QuantizedTensor> init_quantized_tensors(void **ptr, int n, int size_each) const;

        void dequantize(const QuantizedTensor *qx, float *x, int n) const;

        void quantize(const QuantizedTensor *qx, const float *x, int n) const;

        static void rmsnorm(float *o, const float *x, const float *weight, int size);

        static void softmax(float *x, int size);

        void matmul(float *xout, const QuantizedTensor *x, const QuantizedTensor *w, int n, int d) const;
    };
} // namespace Inference

#endif // LEAP_QUANTIZED_TRANSFORMER_H
