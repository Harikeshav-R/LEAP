#include "FloatTransformer.h"
#include <cmath>
#include <cstring>
#include <unistd.h>
#include <sys/mman.h>
#include <stdexcept>
#include <iostream>

namespace Inference {
    FloatTransformer::FloatTransformer(const Config &config, const int fd, void *mmap_ptr, const size_t file_size,
                                       float *weights_ptr,
                                       const int shared_weights)
        : fd(fd), mmap_ptr(mmap_ptr), file_size(file_size) {
        this->config = config;
        memory_map_weights(weights_ptr, shared_weights);
        init_run_state();
    }

    FloatTransformer::~FloatTransformer() {
        if (mmap_ptr != MAP_FAILED) {
            munmap(mmap_ptr, file_size);
        }
        if (fd != -1) {
            close(fd);
        }
    }

    void FloatTransformer::init_run_state() {
        const Config *p = &config;
        const int kv_dim = (p->dim * p->n_kv_heads) / p->n_heads;
        state.x.resize(p->dim);
        state.xb.resize(p->dim);
        state.xb2.resize(p->dim);
        state.hb.resize(p->hidden_dim);
        state.hb2.resize(p->hidden_dim);
        state.q.resize(p->dim);
        state.key_cache.resize(p->n_layers * p->seq_len * kv_dim);
        state.value_cache.resize(p->n_layers * p->seq_len * kv_dim);
        state.att.resize(p->n_heads * p->seq_len);
        state.logits.resize(p->vocab_size);
    }

    void FloatTransformer::rmsnorm(float *o, const float *x, const float *weight, const int size) {
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

    void FloatTransformer::softmax(float *x, const int size) {
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

    void FloatTransformer::matmul(float *xout, const float *x, const float *w, const int n, const int d) {
        // W (d,n) @ x (n,) -> xout (d,)
        int i;
#pragma omp parallel for private(i)
        for (i = 0; i < d; i++) {
            float val = 0.0f;
            for (int j = 0; j < n; j++) {
                val += w[i * n + j] * x[j];
            }
            xout[i] = val;
        }
    }

    float *FloatTransformer::forward(const int token, const int pos) {
        const Config *p = &config;
        const FloatTransformerWeights *w = &weights;
        FloatRunState *s = &state;
        float *x = s->x.data();
        const int dim = p->dim;
        const int kv_dim = (p->dim * p->n_kv_heads) / p->n_heads;
        const int kv_mul = p->n_heads / p->n_kv_heads;
        const int hidden_dim = p->hidden_dim;
        const int head_size = dim / p->n_heads;

        const float *content_row = w->token_embedding_table + token * dim;
        std::memcpy(x, content_row, dim * sizeof(*x));

        for (int l = 0; l < p->n_layers; l++) {
            rmsnorm(s->xb.data(), x, w->rms_att_weight + l * dim, dim);

            const int loff = l * p->seq_len * kv_dim;
            float *k = s->key_cache.data() + loff + pos * kv_dim;
            float *v = s->value_cache.data() + loff + pos * kv_dim;

            matmul(s->q.data(), s->xb.data(), w->wq + l * dim * dim, dim, dim);
            matmul(k, s->xb.data(), w->wk + l * dim * kv_dim, dim, kv_dim);
            matmul(v, s->xb.data(), w->wv + l * dim * kv_dim, dim, kv_dim);

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
                        const float k0 = k[i * head_size + j];
                        const float k1 = k[i * head_size + j + 1];
                        k[i * head_size + j] = k0 * fcr - k1 * fci;
                        k[i * head_size + j + 1] = k0 * fci + k1 * fcr;
                    }
                }
            }

            int h;
#pragma omp parallel for private(h)
            for (h = 0; h < p->n_heads; h++) {
                const float *q = s->q.data() + h * head_size;
                float *att = s->att.data() + h * p->seq_len;
                for (int t = 0; t <= pos; t++) {
                    const float *k_ptr = s->key_cache.data() + loff + t * kv_dim + (h / kv_mul) * head_size;
                    float score = 0.0f;
                    for (int i = 0; i < head_size; i++) {
                        score += q[i] * k_ptr[i];
                    }
                    score /= std::sqrt(static_cast<float>(head_size));
                    att[t] = score;
                }

                softmax(att, pos + 1);

                float *xb = s->xb.data() + h * head_size;
                std::memset(xb, 0, head_size * sizeof(float));
                for (int t = 0; t <= pos; t++) {
                    const float *v_ptr = s->value_cache.data() + loff + t * kv_dim + (h / kv_mul) * head_size;
                    const float a = att[t];
                    for (int i = 0; i < head_size; i++) {
                        xb[i] += a * v_ptr[i];
                    }
                }
            }

            matmul(s->xb2.data(), s->xb.data(), w->wo + l * dim * dim, dim, dim);

            for (int i = 0; i < dim; i++) {
                x[i] += s->xb2[i];
            }

            rmsnorm(s->xb.data(), x, w->rms_ffn_weight + l * dim, dim);

            matmul(s->hb.data(), s->xb.data(), w->w1 + l * dim * hidden_dim, dim, hidden_dim);
            matmul(s->hb2.data(), s->xb.data(), w->w3 + l * dim * hidden_dim, dim, hidden_dim);

            for (int i = 0; i < hidden_dim; i++) {
                float val = s->hb[i];
                val *= (1.0f / (1.0f + std::exp(-val)));
                val *= s->hb2[i];
                s->hb[i] = val;
            }

            matmul(s->xb.data(), s->hb.data(), w->w2 + l * dim * hidden_dim, hidden_dim, dim);

            for (int i = 0; i < dim; i++) {
                x[i] += s->xb[i];
            }
        }

        rmsnorm(x, x, w->rms_final_weight, dim);
        matmul(s->logits.data(), x, w->wcls, dim, p->vocab_size);
        return s->logits.data();
    }

    void FloatTransformer::memory_map_weights(float *ptr, const int shared_weights) {
        const int head_size = config.dim / config.n_heads;
        const unsigned long long n_layers = config.n_layers;

        weights.rms_att_weight = ptr;
        ptr += n_layers * config.dim;
        weights.rms_ffn_weight = ptr;
        ptr += n_layers * config.dim;
        weights.rms_final_weight = ptr;
        ptr += config.dim;
        weights.token_embedding_table = ptr;
        ptr += config.vocab_size * config.dim;
        weights.wq = ptr;
        ptr += n_layers * config.dim * (config.n_heads * head_size);
        weights.wk = ptr;
        ptr += n_layers * config.dim * (config.n_kv_heads * head_size);
        weights.wv = ptr;
        ptr += n_layers * config.dim * (config.n_kv_heads * head_size);
        weights.wo = ptr;
        ptr += n_layers * (config.n_heads * head_size) * config.dim;
        weights.w1 = ptr;
        ptr += n_layers * config.dim * config.hidden_dim;
        weights.w2 = ptr;
        ptr += n_layers * config.hidden_dim * config.dim;
        weights.w3 = ptr;
        ptr += n_layers * config.dim * config.hidden_dim;
        if (shared_weights) {
            weights.wcls = weights.token_embedding_table;
        } else {
            weights.wcls = ptr;
        }
    }
} // namespace Inference