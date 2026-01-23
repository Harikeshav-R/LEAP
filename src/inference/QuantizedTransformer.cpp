#include "QuantizedTransformer.h"
#include <cmath>
#include <unistd.h>
#include <sys/mman.h>
#include <stdexcept>
#include <iostream>

namespace Inference {
    QuantizedTransformer::QuantizedTransformer(const Config &config, const int fd, void *mmap_ptr,
                                               const ssize_t file_size, void *weights_ptr, const int shared_weights,
                                               const int group_size)
        : fd(fd), mmap_ptr(mmap_ptr), file_size(file_size), group_size(group_size) {
        this->config = config;
        memory_map_weights(weights_ptr, shared_weights);
        malloc_run_state();
    }

    QuantizedTransformer::~QuantizedTransformer() {
        free_run_state();
        // free QuantizedTensors allocated in memory_map_weights (actually the struct arrays are malloced)
        free(weights.q_tokens);
        free(weights.token_embedding_table);
        free(weights.wq);
        free(weights.wk);
        free(weights.wv);
        free(weights.wo);
        free(weights.w1);
        free(weights.w2);
        free(weights.w3);
        if (weights.wcls != weights.q_tokens) {
            free(weights.wcls);
        }

        if (mmap_ptr != MAP_FAILED) {
            munmap(mmap_ptr, file_size);
        }
        if (fd != -1) {
            close(fd);
        }
    }

    void QuantizedTransformer::malloc_run_state() {
        Config *p = &config;
        const int kv_dim = (p->dim * p->n_kv_heads) / p->n_heads;
        state.x = static_cast<float *>(calloc(p->dim, sizeof(float)));
        state.xb = static_cast<float *>(calloc(p->dim, sizeof(float)));
        state.xb2 = static_cast<float *>(calloc(p->dim, sizeof(float)));
        state.hb = static_cast<float *>(calloc(p->hidden_dim, sizeof(float)));
        state.hb2 = static_cast<float *>(calloc(p->hidden_dim, sizeof(float)));

        state.xq.q = static_cast<int8_t *>(calloc(p->dim, sizeof(int8_t)));
        state.xq.s = static_cast<float *>(calloc(p->dim, sizeof(float)));
        // Wait, s size depends on GS? No, quantize writes to s buffer.
        // In runq.c: s->xq = (QuantizedTensor){.q = calloc(p->dim, sizeof(int8_t)), .s = calloc(p->dim, sizeof(float))};
        // Wait, the scaling factor is per group?
        // runq.c: qx->s[group] = scale;
        // But allocation is `calloc(p->dim, sizeof(float))`.
        // Since GS >= 1, allocating `dim` floats is safe (over-allocated but safe).

        state.hq.q = static_cast<int8_t *>(calloc(p->hidden_dim, sizeof(int8_t)));
        state.hq.s = static_cast<float *>(calloc(p->hidden_dim, sizeof(float)));

        state.q = static_cast<float *>(calloc(p->dim, sizeof(float)));
        state.k = static_cast<float *>(calloc(kv_dim, sizeof(float)));
        state.v = static_cast<float *>(calloc(kv_dim, sizeof(float)));
        state.att = static_cast<float *>(calloc(p->n_heads * p->seq_len, sizeof(float)));
        state.logits = static_cast<float *>(calloc(p->vocab_size, sizeof(float)));
        state.key_cache = static_cast<float *>(calloc(p->n_layers * p->seq_len * kv_dim, sizeof(float)));
        state.value_cache = static_cast<float *>(calloc(p->n_layers * p->seq_len * kv_dim, sizeof(float)));
    }

    void QuantizedTransformer::free_run_state() const {
        free(state.x);
        free(state.xb);
        free(state.xb2);
        free(state.hb);
        free(state.hb2);
        free(state.xq.q);
        free(state.xq.s);
        free(state.hq.q);
        free(state.hq.s);
        free(state.q);
        free(state.k);
        free(state.v);
        free(state.att);
        free(state.logits);
        free(state.key_cache);
        free(state.value_cache);
    }

    QuantizedTensor *QuantizedTransformer::init_quantized_tensors(void **ptr, int n, int size_each) const {
        void *p = *ptr;
        auto *res = static_cast<QuantizedTensor *>(malloc(n * sizeof(QuantizedTensor)));
        for (int i = 0; i < n; i++) {
            res[i].q = static_cast<int8_t *>(p);
            p = static_cast<int8_t *>(p) + size_each;
            res[i].s = static_cast<float *>(p);
            p = static_cast<float *>(p) + size_each / group_size;
        }
        *ptr = p;
        return res;
    }

    void QuantizedTransformer::memory_map_weights(void *ptr, int shared_weights) {
        const int head_size = config.dim / config.n_heads;

        auto *fptr = static_cast<float *>(ptr);
        weights.rms_att_weight = fptr;
        fptr += config.n_layers * config.dim;
        weights.rms_ffn_weight = fptr;
        fptr += config.n_layers * config.dim;
        weights.rms_final_weight = fptr;
        fptr += config.dim;

        ptr = static_cast<void *>(fptr);
        weights.q_tokens = init_quantized_tensors(&ptr, 1, config.vocab_size * config.dim);

        weights.token_embedding_table = static_cast<float *>(malloc(config.vocab_size * config.dim * sizeof(float)));
        dequantize(weights.q_tokens, weights.token_embedding_table, config.vocab_size * config.dim);

        weights.wq = init_quantized_tensors(&ptr, config.n_layers, config.dim * (config.n_heads * head_size));
        weights.wk = init_quantized_tensors(&ptr, config.n_layers, config.dim * (config.n_kv_heads * head_size));
        weights.wv = init_quantized_tensors(&ptr, config.n_layers, config.dim * (config.n_kv_heads * head_size));
        weights.wo = init_quantized_tensors(&ptr, config.n_layers, (config.n_heads * head_size) * config.dim);

        weights.w1 = init_quantized_tensors(&ptr, config.n_layers, config.dim * config.hidden_dim);
        weights.w2 = init_quantized_tensors(&ptr, config.n_layers, config.hidden_dim * config.dim);
        weights.w3 = init_quantized_tensors(&ptr, config.n_layers, config.dim * config.hidden_dim);

        weights.wcls = shared_weights
                           ? weights.q_tokens
                           : init_quantized_tensors(&ptr, 1, config.dim * config.vocab_size);
    }

    void QuantizedTransformer::dequantize(const QuantizedTensor *qx, float *x, const int n) const {
        for (int i = 0; i < n; i++) {
            x[i] = qx->q[i] * qx->s[i / group_size];
        }
    }

    void QuantizedTransformer::quantize(const QuantizedTensor *qx, const float *x, const int n) const {
        const int num_groups = n / group_size;

        for (int group = 0; group < num_groups; group++) {
            constexpr float Q_MAX = 127.0f;
            float wmax = 0.0f;
            for (int i = 0; i < group_size; i++) {
                if (const float val = fabs(x[group * group_size + i]); val > wmax) wmax = val;
            }

            const float scale = wmax / Q_MAX;
            qx->s[group] = scale;

            for (int i = 0; i < group_size; i++) {
                const float quant_value = x[group * group_size + i] / scale;
                const auto quantized = static_cast<int8_t>(round(quant_value));
                qx->q[group * group_size + i] = quantized;
            }
        }
    }

    void QuantizedTransformer::rmsnorm(float *o, const float *x, const float *weight, const int size) {
        float ss = 0.0f;
        for (int j = 0; j < size; j++) {
            ss += x[j] * x[j];
        }
        ss /= size;
        ss += 1e-5f;
        ss = 1.0f / sqrtf(ss);
        for (int j = 0; j < size; j++) {
            o[j] = weight[j] * (ss * x[j]);
        }
    }

    void QuantizedTransformer::softmax(float *x, const int size) {
        float max_val = x[0];
        for (int i = 1; i < size; i++) {
            if (x[i] > max_val) max_val = x[i];
        }
        float sum = 0.0f;
        for (int i = 0; i < size; i++) {
            x[i] = expf(x[i] - max_val);
            sum += x[i];
        }
        for (int i = 0; i < size; i++) {
            x[i] /= sum;
        }
    }

    void QuantizedTransformer::matmul(float *xout, const QuantizedTensor *x, const QuantizedTensor *w, const int n,
                                      const int d) const {
        int i;
#pragma omp parallel for private(i)
        for (i = 0; i < d; i++) {
            float val = 0.0f;
            int32_t ival = 0;
            const int in = i * n;

            for (int j = 0; j <= n - group_size; j += group_size) {
                for (int k = 0; k < group_size; k++) {
                    ival += static_cast<int32_t>(x->q[j + k]) * static_cast<int32_t>(w->q[in + j + k]);
                }
                val += static_cast<float>(ival) * w->s[(in + j) / group_size] * x->s[j / group_size];
                ival = 0;
            }
            xout[i] = val;
        }
    }

    float *QuantizedTransformer::forward(int token, int pos) {
        const Config *p = &config;
        const QuantizedTransformerWeights *w = &weights;
        const QuantizedRunState *s = &state;
        float *x = s->x;
        const int dim = p->dim;
        const int kv_dim = (p->dim * p->n_kv_heads) / p->n_heads;
        const int kv_mul = p->n_heads / p->n_kv_heads;
        const int hidden_dim = p->hidden_dim;
        const int head_size = dim / p->n_heads;

        memcpy(x, w->token_embedding_table + token * dim, dim * sizeof(float));

        for (int l = 0; l < p->n_layers; l++) {
            rmsnorm(s->xb, x, w->rms_att_weight + l * dim, dim);

            quantize(&s->xq, s->xb, dim);
            matmul(s->q, &s->xq, w->wq + l, dim, dim);
            matmul(s->k, &s->xq, w->wk + l, dim, kv_dim);
            matmul(s->v, &s->xq, w->wv + l, dim, kv_dim);

            for (int i = 0; i < p->n_heads; i++) {
                for (int j = 0; j < head_size; j += 2) {
                    const float freq = 1.0f / powf(500000.0f, static_cast<float>(j) / static_cast<float>(head_size));
                    const float val = pos * freq;
                    const float fcr = cosf(val);
                    const float fci = sinf(val);
                    const float q0 = s->q[i * head_size + j];
                    const float q1 = s->q[i * head_size + j + 1];
                    s->q[i * head_size + j] = q0 * fcr - q1 * fci;
                    s->q[i * head_size + j + 1] = q0 * fci + q1 * fcr;
                    if (i < p->n_kv_heads) {
                        const float k0 = s->k[i * head_size + j];
                        const float k1 = s->k[i * head_size + j + 1];
                        s->k[i * head_size + j] = k0 * fcr - k1 * fci;
                        s->k[i * head_size + j + 1] = k0 * fci + k1 * fcr;
                    }
                }
            }

            const int loff = l * p->seq_len * kv_dim;
            float *key_cache_row = s->key_cache + loff + pos * kv_dim;
            float *value_cache_row = s->value_cache + loff + pos * kv_dim;
            memcpy(key_cache_row, s->k, kv_dim * sizeof(*key_cache_row));
            memcpy(value_cache_row, s->v, kv_dim * sizeof(*value_cache_row));

            int h;
#pragma omp parallel for private(h)
            for (h = 0; h < p->n_heads; h++) {
                float *q = s->q + h * head_size;
                float *att = s->att + h * p->seq_len;
                for (int t = 0; t <= pos; t++) {
                    float *k = s->key_cache + loff + t * kv_dim + (h / kv_mul) * head_size;
                    float score = 0.0f;
                    for (int i = 0; i < head_size; i++) {
                        score += q[i] * k[i];
                    }
                    score /= sqrtf(head_size);
                    att[t] = score;
                }

                softmax(att, pos + 1);

                float *xb = s->xb + h * head_size;
                memset(xb, 0, head_size * sizeof(float));
                for (int t = 0; t <= pos; t++) {
                    float *v = s->value_cache + loff + t * kv_dim + (h / kv_mul) * head_size;
                    float a = att[t];
                    for (int i = 0; i < head_size; i++) {
                        xb[i] += a * v[i];
                    }
                }
            }

            quantize(&s->xq, s->xb, dim);
            matmul(s->xb2, &s->xq, w->wo + l, dim, dim);

            for (int i = 0; i < dim; i++) {
                x[i] += s->xb2[i];
            }

            rmsnorm(s->xb, x, w->rms_ffn_weight + l * dim, dim);

            quantize(&s->xq, s->xb, dim);
            matmul(s->hb, &s->xq, w->w1 + l, dim, hidden_dim);
            matmul(s->hb2, &s->xq, w->w3 + l, dim, hidden_dim);

            for (int i = 0; i < hidden_dim; i++) {
                float val = s->hb[i];
                val *= (1.0f / (1.0f + expf(-val)));
                val *= s->hb2[i];
                s->hb[i] = val;
            }

            quantize(&s->hq, s->hb, hidden_dim);
            matmul(s->xb, &s->hq, w->w2 + l, hidden_dim, dim);

            for (int i = 0; i < dim; i++) {
                x[i] += s->xb[i];
            }
        }

        rmsnorm(x, x, w->rms_final_weight, dim);
        quantize(&s->xq, x, dim);
        matmul(s->logits, &s->xq, w->wcls, dim, p->vocab_size);
        return s->logits;
    }
} // namespace Inference