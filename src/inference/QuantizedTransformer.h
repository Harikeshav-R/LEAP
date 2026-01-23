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
        QuantizedTensor *q_tokens; // (vocab_size, dim)
        float *token_embedding_table; // same, but dequantized

        float *rms_att_weight;
        float *rms_ffn_weight;

        QuantizedTensor *wq;
        QuantizedTensor *wk;
        QuantizedTensor *wv;
        QuantizedTensor *wo;

        QuantizedTensor *w1;
        QuantizedTensor *w2;
        QuantizedTensor *w3;

        float *rms_final_weight;
        QuantizedTensor *wcls;
    };

    struct QuantizedRunState {
        float *x;
        float *xb;
        float *xb2;
        float *hb;
        float *hb2;
        QuantizedTensor xq;
        QuantizedTensor hq;
        float *q;
        float *k;
        float *v;
        float *att;
        float *logits;
        float *key_cache;
        float *value_cache;
    };

    class QuantizedTransformer : public Transformer {
    public:
        QuantizedTransformer(const Config &config, int fd, void *mmap_ptr, ssize_t file_size, void *weights_ptr,
                             int shared_weights, int group_size);

        ~QuantizedTransformer() override;

        float *forward(int token, int pos) override;

    private:
        QuantizedTransformerWeights weights{};
        QuantizedRunState state{};
        int fd;
        void *mmap_ptr;
        ssize_t file_size;
        int group_size; // GS

        void memory_map_weights(void *ptr, int shared_weights);

        void malloc_run_state();

        void free_run_state() const;

        // Helpers
        QuantizedTensor *init_quantized_tensors(void **ptr, int n, int size_each) const;

        void dequantize(const QuantizedTensor *qx, float *x, int n) const;

        void quantize(const QuantizedTensor *qx, const float *x, int n) const;

        static void rmsnorm(float *o, const float *x, const float *weight, int size);

        static void softmax(float *x, int size);

        void matmul(float *xout, const QuantizedTensor *x, const QuantizedTensor *w, int n, int d) const;
    };
} // namespace Inference

#endif // LEAP_QUANTIZED_TRANSFORMER_H