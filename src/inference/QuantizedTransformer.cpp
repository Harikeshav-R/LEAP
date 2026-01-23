#include "QuantizedTransformer.h"
#include <cmath>
#include <unistd.h>
#include <sys/mman.h>
#include <stdexcept>
#include <iostream>
#include <vector>

namespace Inference {
    QuantizedTransformer::QuantizedTransformer(const Config &config, const int fd, void *mmap_ptr,
                                               const size_t file_size, void *weights_ptr, const int shared_weights,
                                               const int group_size)
        : fd(fd), mmap_ptr(mmap_ptr), file_size(file_size), group_size(group_size) {
        this->config = config;
        memory_map_weights(weights_ptr, shared_weights);
        init_run_state();
    }

    QuantizedTransformer::~QuantizedTransformer() {
        // Free weights that were allocated with new[]
        delete[] weights.q_tokens;
        delete[] weights.token_embedding_table;
        delete[] weights.wq;
        delete[] weights.wk;
        delete[] weights.wv;
        delete[] weights.wo;
        delete[] weights.w1;
        delete[] weights.w2;
        delete[] weights.w3;

        if (weights.wcls != weights.q_tokens) {
            delete[] weights.wcls;
        }

        if (mmap_ptr != MAP_FAILED) {
            munmap(mmap_ptr, file_size);
        }
        if (fd != -1) {
            close(fd);
        }
    }

    void QuantizedTransformer::init_run_state() {
        const Config *p = &config;
        const int kv_dim = (p->dim * p->n_kv_heads) / p->n_heads;

        state.x.resize(p->dim);
        state.xb.resize(p->dim);
        state.xb2.resize(p->dim);
        state.hb.resize(p->hidden_dim);
        state.hb2.resize(p->hidden_dim);

        state.xq_q.resize(p->dim);
        state.xq_s.resize(p->dim); // Safety size

        state.hq_q.resize(p->hidden_dim);
        state.hq_s.resize(p->hidden_dim);

        state.q.resize(p->dim);
        state.k.resize(kv_dim);
        state.v.resize(kv_dim);
        state.att.resize(p->n_heads * p->seq_len);
        state.logits.resize(p->vocab_size);
        state.key_cache.resize(p->n_layers * p->seq_len * kv_dim);
        state.value_cache.resize(p->n_layers * p->seq_len * kv_dim);
    }

    QuantizedTensor *QuantizedTransformer::init_quantized_tensors(void **ptr, const int n, const int size_each) const {
        void *p = *ptr;
        auto *res = new QuantizedTensor[n];
        for (int i = 0; i < n; i++) {
            res[i].q = static_cast<int8_t *>(p);
            p = static_cast<int8_t *>(p) + size_each;
            res[i].s = static_cast<float *>(p);
            p = static_cast<float *>(p) + size_each / group_size;
        }
        *ptr = p;
        return res;
    }

    void QuantizedTransformer::memory_map_weights(void *ptr, const int shared_weights) {
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

        weights.token_embedding_table = new float[config.vocab_size * config.dim];
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
                if (const float val = std::fabs(x[group * group_size + i]); val > wmax) wmax = val;
            }

            const float scale = wmax / Q_MAX;
            qx->s[group] = scale;

            for (int i = 0; i < group_size; i++) {
                const float quant_value = x[group * group_size + i] / scale;
                const auto quantized = static_cast<int8_t>(std::round(quant_value));
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
        ss = 1.0f / std::sqrt(ss);
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
            x[i] = std::exp(x[i] - max_val);
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

    float *QuantizedTransformer::forward(const int token, const int pos) {
        const Config *p = &config;
        const QuantizedTransformerWeights *w = &weights;
        QuantizedRunState *s = &state;
        float *x = s->x.data();
        const int dim = p->dim;
        const int kv_dim = (p->dim * p->n_kv_heads) / p->n_heads;
        const int kv_mul = p->n_heads / p->n_kv_heads;
        const int hidden_dim = p->hidden_dim;
        const int head_size = dim / p->n_heads;

        std::memcpy(x, w->token_embedding_table + token * dim, dim * sizeof(float));

        for (int l = 0; l < p->n_layers; l++) {
            rmsnorm(s->xb.data(), x, w->rms_att_weight + l * dim, dim);

            // Construct temp QuantizedTensor for xq
            QuantizedTensor xq_tensor{s->xq_q.data(), s->xq_s.data()};
            quantize(&xq_tensor, s->xb.data(), dim);

            matmul(s->q.data(), &xq_tensor, w->wq + l, dim, dim);
            matmul(s->k.data(), &xq_tensor, w->wk + l, dim, kv_dim);
            matmul(s->v.data(), &xq_tensor, w->wv + l, dim, kv_dim);

            for (int i = 0; i < p->n_heads; i++) {
                for (int j = 0; j < head_size; j += 2) {
                    const float freq =
                            1.0f / std::pow(500000.0f, static_cast<float>(j) / static_cast<float>(head_size));
                    const float val = pos * freq;
                    const float fcr = std::cos(val);
                    const float fci = std::sin(val);
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
            float *key_cache_row = s->key_cache.data() + loff + pos * kv_dim;
            float *value_cache_row = s->value_cache.data() + loff + pos * kv_dim;
            std::memcpy(key_cache_row, s->k.data(), kv_dim * sizeof(*key_cache_row));
            std::memcpy(value_cache_row, s->v.data(), kv_dim * sizeof(*value_cache_row));

            int h;
#pragma omp parallel for private(h)
            for (h = 0; h < p->n_heads; h++) {
                const float *q = s->q.data() + h * head_size;
                float *att = s->att.data() + h * p->seq_len;
                for (int t = 0; t <= pos; t++) {
                    const float *k = s->key_cache.data() + loff + t * kv_dim + (h / kv_mul) * head_size;
                    float score = 0.0f;
                    for (int i = 0; i < head_size; i++) {
                        score += q[i] * k[i];
                    }
                    score /= std::sqrt(static_cast<float>(head_size));
                    att[t] = score;
                }

                softmax(att, pos + 1);

                float *xb = s->xb.data() + h * head_size;
                std::memset(xb, 0, head_size * sizeof(float));
                for (int t = 0; t <= pos; t++) {
                    const float *v = s->value_cache.data() + loff + t * kv_dim + (h / kv_mul) * head_size;
                    const float a = att[t];
                    for (int i = 0; i < head_size; i++) {
                        xb[i] += a * v[i];
                    }
                }
            }

            quantize(&xq_tensor, s->xb.data(), dim);
            matmul(s->xb2.data(), &xq_tensor, w->wo + l, dim, dim);

            for (int i = 0; i < dim; i++) {
                x[i] += s->xb2[i];
            }

            rmsnorm(s->xb.data(), x, w->rms_ffn_weight + l * dim, dim);

            quantize(&xq_tensor, s->xb.data(), dim);
            matmul(s->hb.data(), &xq_tensor, w->w1 + l, dim, hidden_dim);
            matmul(s->hb2.data(), &xq_tensor, w->w3 + l, dim, hidden_dim);

            for (int i = 0; i < hidden_dim; i++) {
                float val = s->hb[i];
                val *= (1.0f / (1.0f + std::exp(-val)));
                val *= s->hb2[i];
                s->hb[i] = val;
            }

            QuantizedTensor hq_tensor{s->hq_q.data(), s->hq_s.data()};
            quantize(&hq_tensor, s->hb.data(), hidden_dim);
            matmul(s->xb.data(), &hq_tensor, w->w2 + l, hidden_dim, dim);

            for (int i = 0; i < dim; i++) {
                x[i] += s->xb[i];
            }
        }

        rmsnorm(x, x, w->rms_final_weight, dim);
        const QuantizedTensor xq_tensor{s->xq_q.data(), s->xq_s.data()};
        quantize(&xq_tensor, x, dim);
        matmul(s->logits.data(), &xq_tensor, w->wcls, dim, p->vocab_size);
        return s->logits.data();
    }
} // namespace Inference