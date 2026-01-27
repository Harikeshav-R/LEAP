#include "FloatTransformer.h"
#include <cmath>
#include <unistd.h>
#include <sys/mman.h>
#include <stdexcept>
#include <iostream>
#include <algorithm>
#include <cstring>

#if defined(__ARM_NEON)
#include <arm_neon.h>
#endif

namespace Inference {
    FloatTransformer::FloatTransformer(const Config &config, const int fd, void *mmap_ptr, const size_t file_size,
                                       float *weights_ptr,
                                       const int shared_weights)
        : fd(fd), mmap_ptr(mmap_ptr), file_size(file_size) {
        this->config = config;
        memory_map_weights(weights_ptr, shared_weights);
        init_run_state();
        precompute_freqs();
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

    void FloatTransformer::precompute_freqs() {
        const int head_size = config.dim / config.n_heads;
        const int seq_len = config.seq_len;

        rope_cos.resize(seq_len * (head_size / 2));
        rope_sin.resize(seq_len * (head_size / 2));

        for (int pos = 0; pos < seq_len; pos++) {
            for (int i = 0; i < head_size; i += 2) {
                const float freq = 1.0f / std::pow(500000.0f, static_cast<float>(i) / static_cast<float>(head_size));
                const float val = static_cast<float>(pos) * freq;
                const int index = pos * (head_size / 2) + (i / 2);
                rope_cos[index] = std::cos(val);
                rope_sin[index] = std::sin(val);
            }
        }
    }

    void FloatTransformer::rmsnorm(float *__restrict__ o, const float *__restrict__ x, const float *__restrict__ weight,
                                   const int size) {
        float ss = 0.0f;
#if defined(__ARM_NEON)
        float32x4_t sum_vec = vdupq_n_f32(0.0f);
        int j = 0;
        for (; j <= size - 4; j += 4) {
            float32x4_t val = vld1q_f32(x + j);
            sum_vec = vmlaq_f32(sum_vec, val, val);
        }
        ss = vaddvq_f32(sum_vec);
        for (; j < size; j++) {
            ss += x[j] * x[j];
        }
#else
        for (int j = 0; j < size; j++) {
            ss += x[j] * x[j];
        }
#endif
        ss /= static_cast<float>(size);
        ss += 1e-5f;
        ss = 1.0f / std::sqrt(ss);

#if defined(__ARM_NEON)
        j = 0;
        float32x4_t ss_vec = vdupq_n_f32(ss);
        for (; j <= size - 4; j += 4) {
            float32x4_t w_val = vld1q_f32(weight + j);
            float32x4_t x_val = vld1q_f32(x + j);
            float32x4_t res = vmulq_f32(w_val, vmulq_f32(ss_vec, x_val));
            vst1q_f32(o + j, res);
        }
        for (; j < size; j++) {
            o[j] = weight[j] * (ss * x[j]);
        }
#else
        for (int j = 0; j < size; j++) {
            o[j] = weight[j] * (ss * x[j]);
        }
#endif
    }

    void FloatTransformer::softmax(float *x, const int size) {
        const float max_val = *std::max_element(x, x + size);
        float sum = 0.0f;
        for (int i = 0; i < size; i++) {
            x[i] = std::exp(x[i] - max_val);
            sum += x[i];
        }
        const float inv_sum = 1.0f / sum;
        for (int i = 0; i < size; i++) {
            x[i] *= inv_sum;
        }
    }

    void FloatTransformer::matmul(float *__restrict__ xout, const float *__restrict__ x, const float *__restrict__ w,
                                  const int n, const int d) {
        // W (d,n) @ x (n,) -> xout (d,)
        int i;
#pragma omp parallel for private(i) schedule(static)
        for (i = 0; i < d; i += 4) {
            if (i + 3 < d) {
                float val0 = 0.0f;
                float val1 = 0.0f;
                float val2 = 0.0f;
                float val3 = 0.0f;
                int j = 0;
                const int offset0 = i * n;
                const int offset1 = (i + 1) * n;
                const int offset2 = (i + 2) * n;
                const int offset3 = (i + 3) * n;

#if defined(__ARM_NEON)
                float32x4_t sum0 = vdupq_n_f32(0.0f);
                float32x4_t sum1 = vdupq_n_f32(0.0f);
                float32x4_t sum2 = vdupq_n_f32(0.0f);
                float32x4_t sum3 = vdupq_n_f32(0.0f);

                for (; j <= n - 4; j += 4) {
                    float32x4_t x_vec = vld1q_f32(x + j);

                    float32x4_t w0 = vld1q_f32(w + offset0 + j);
                    float32x4_t w1 = vld1q_f32(w + offset1 + j);
                    float32x4_t w2 = vld1q_f32(w + offset2 + j);
                    float32x4_t w3 = vld1q_f32(w + offset3 + j);

                    sum0 = vmlaq_f32(sum0, w0, x_vec);
                    sum1 = vmlaq_f32(sum1, w1, x_vec);
                    sum2 = vmlaq_f32(sum2, w2, x_vec);
                    sum3 = vmlaq_f32(sum3, w3, x_vec);
                }
                val0 = vaddvq_f32(sum0);
                val1 = vaddvq_f32(sum1);
                val2 = vaddvq_f32(sum2);
                val3 = vaddvq_f32(sum3);
#endif
                for (; j < n; j++) {
                    float xv = x[j];
                    val0 += w[offset0 + j] * xv;
                    val1 += w[offset1 + j] * xv;
                    val2 += w[offset2 + j] * xv;
                    val3 += w[offset3 + j] * xv;
                }
                xout[i] = val0;
                xout[i + 1] = val1;
                xout[i + 2] = val2;
                xout[i + 3] = val3;
            } else {
                for (int k = i; k < d; k++) {
                    float val = 0.0f;
                    int j = 0;
                    const int offset = k * n;
#if defined(__ARM_NEON)
                    float32x4_t sum_vec = vdupq_n_f32(0.0f);
                    for (; j <= n - 4; j += 4) {
                        float32x4_t w_vec = vld1q_f32(w + offset + j);
                        float32x4_t x_vec = vld1q_f32(x + j);
                        sum_vec = vmlaq_f32(sum_vec, w_vec, x_vec);
                    }
                    val = vaddvq_f32(sum_vec);
#endif
                    for (; j < n; j++) {
                        val += w[offset + j] * x[j];
                    }
                    xout[k] = val;
                }
            }
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
        std::copy_n(content_row, dim, x);

        for (int l = 0; l < p->n_layers; l++) {
            rmsnorm(s->xb.data(), x, w->rms_att_weight + l * dim, dim);

            const int loff = l * p->seq_len * kv_dim;
            float *k = s->key_cache.data() + loff + pos * kv_dim;
            float *v = s->value_cache.data() + loff + pos * kv_dim;

            matmul(s->q.data(), s->xb.data(), w->wq + l * dim * dim, dim, dim);
            matmul(k, s->xb.data(), w->wk + l * dim * kv_dim, dim, kv_dim);
            matmul(v, s->xb.data(), w->wv + l * dim * kv_dim, dim, kv_dim);

            // RoPE
            const int rope_offset = pos * (head_size / 2);
            for (int i = 0; i < p->n_heads; i++) {
                int j = 0;
#if defined(__ARM_NEON)
                for (; j <= head_size - 4; j += 4) {
                    // Load 4 floats: [q0, q1, q2, q3] -> 2 pairs (q0,q1), (q2,q3)
                    float32x4_t q_vec = vld1q_f32(s->q.data() + i * head_size + j);

                    // Load cos/sin. rope_cos is packed as [cos0, cos1, ...] for each pair.
                    // We need [cos0, cos0, cos1, cos1]
                    // The rope_cos array has 1 value per pair.
                    // We load 2 values: cos0, cos1.
                    // We need to duplicate them.

                    float c0 = rope_cos[rope_offset + j / 2];
                    float c1 = rope_cos[rope_offset + j / 2 + 1];
                    float s0 = rope_sin[rope_offset + j / 2];
                    float s1 = rope_sin[rope_offset + j / 2 + 1];

                    // Construct vectors: [c0, c0, c1, c1]
                    float32x4_t c_vec = {c0, c0, c1, c1};
                    float32x4_t s_vec = {s0, s0, s1, s1};

                    // Create [-s, c, -s, c] ? No.
                    // q_out_0 = q0*c0 - q1*s0
                    // q_out_1 = q0*s0 + q1*c0

                    // Deconstruct q_vec to [q0, q0, q2, q2] and [q1, q1, q3, q3]?
                    // Or standard complex mul logic.
                    // q_real = [q0, q2], q_imag = [q1, q3]
                    // This requires shuffling.

                    // trn1 (transpose) q_vec with itself?
                    // vuzp1/2 (unzip).
                    // q_real (even indices): vuzp1q_f32(q_vec, q_vec) -> [q0, q2, q0, q2]
                    // But standard logic:
                    // rotated = (q * c) + (swapped_q * (-s, s, -s, s))
                    // swapped_q = [-q1, q0, -q3, q2]

                    // Step 1: Compute q * c
                    float32x4_t term1 = vmulq_f32(q_vec, c_vec);

                    // Step 2: Swap q to get [q1, q0, q3, q2]
                    // vrev64q_f32 swaps pairs of 32-bit elements within 64-bit granules.
                    float32x4_t q_swap = vrev64q_f32(q_vec);

                    // Step 3: Compute swapped_q * s
                    float32x4_t term2 = vmulq_f32(q_swap, s_vec);

                    // Step 4: Apply signs.
                    // We want:
                    // term1[0] = q0*c0
                    // term1[1] = q1*c0
                    // term2[0] = q1*s0
                    // term2[1] = q0*s0
                    //
                    // result[0] = q0*c0 - q1*s0
                    // result[1] = q1*c0 + q0*s0
                    //
                    // We can multiply s_vec by [-1, 1, -1, 1] beforehand?
                    // Or just subtract/add.
                    // NEON has vsub/vadd. We need sub for even, add for odd.
                    // Easier: Multiply q_swap by [-s0, s0, -s1, s1].

                    float32x4_t s_signed = {-s0, s0, -s1, s1};
                    term2 = vmulq_f32(q_swap, s_signed);

                    float32x4_t res = vaddq_f32(term1, term2);
                    vst1q_f32(s->q.data() + i * head_size + j, res);

                    if (i < p->n_kv_heads) {
                        float32x4_t k_vec = vld1q_f32(k + i * head_size + j);
                        float32x4_t k_term1 = vmulq_f32(k_vec, c_vec);
                        float32x4_t k_swap = vrev64q_f32(k_vec);
                        float32x4_t k_term2 = vmulq_f32(k_swap, s_signed);
                        float32x4_t k_res = vaddq_f32(k_term1, k_term2);
                        vst1q_f32(k + i * head_size + j, k_res);
                    }
                }
#endif
                for (; j < head_size; j += 2) {
                    const float fcr = rope_cos[rope_offset + j / 2];
                    const float fci = rope_sin[rope_offset + j / 2];

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

                    // Dot product for attention score
                    float score = 0.0f;
                    int i = 0;
#if defined(__ARM_NEON)
                    float32x4_t sum_vec = vdupq_n_f32(0.0f);
                    for (; i <= head_size - 4; i += 4) {
                        float32x4_t q_vec = vld1q_f32(q + i);
                        float32x4_t k_vec = vld1q_f32(k_ptr + i);
                        sum_vec = vmlaq_f32(sum_vec, q_vec, k_vec);
                    }
                    score = vaddvq_f32(sum_vec);
#endif
                    for (; i < head_size; i++) {
                        score += q[i] * k_ptr[i];
                    }
                    score /= std::sqrt(static_cast<float>(head_size));
                    att[t] = score;
                }

                softmax(att, pos + 1);

                float *xb = s->xb.data() + h * head_size;
                std::fill_n(xb, head_size, 0.0f);

                for (int t = 0; t <= pos; t++) {
                    const float *v_ptr = s->value_cache.data() + loff + t * kv_dim + (h / kv_mul) * head_size;
                    const float a = att[t];
                    int i = 0;
#if defined(__ARM_NEON)
                    float32x4_t a_vec = vdupq_n_f32(a);
                    for (; i <= head_size - 4; i += 4) {
                        float32x4_t xb_vec = vld1q_f32(xb + i);
                        float32x4_t v_vec = vld1q_f32(v_ptr + i);
                        xb_vec = vmlaq_f32(xb_vec, a_vec, v_vec);
                        vst1q_f32(xb + i, xb_vec);
                    }
#endif
                    for (; i < head_size; i++) {
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
                val *= (1.0f / (1.0f + std::exp(-val))); // Silu
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