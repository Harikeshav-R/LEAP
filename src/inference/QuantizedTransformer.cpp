#include "QuantizedTransformer.h"
#include <cmath>
#include <cstring>
#include <unistd.h>
#include <sys/mman.h>
#include <stdexcept>
#include <iostream>
#include <vector>
#include <algorithm>

#if defined(__ARM_NEON)
#include <arm_neon.h>
#endif

namespace Inference {
    QuantizedTransformer::QuantizedTransformer(const Config &config, const int fd, void *mmap_ptr,
                                               const size_t file_size, void *weights_ptr, const int shared_weights,
                                               const int group_size)
        : fd(fd), mmap_ptr(mmap_ptr), file_size(file_size), group_size(group_size) {
        this->config = config;
        memory_map_weights(weights_ptr, shared_weights);
        init_run_state();
        precompute_freqs();
    }

    QuantizedTransformer::~QuantizedTransformer() {
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

    void QuantizedTransformer::precompute_freqs() {
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

    std::vector<QuantizedTensor> QuantizedTransformer::init_quantized_tensors(
        void **ptr, const int n, const int size_each) const {
        void *p = *ptr;
        std::vector<QuantizedTensor> res(n);
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

        weights.token_embedding_table.resize(config.vocab_size * config.dim);
        dequantize(&weights.q_tokens[0], weights.token_embedding_table.data(), config.vocab_size * config.dim);

        weights.wq = init_quantized_tensors(&ptr, config.n_layers, config.dim * (config.n_heads * head_size));
        weights.wk = init_quantized_tensors(&ptr, config.n_layers, config.dim * (config.n_kv_heads * head_size));
        weights.wv = init_quantized_tensors(&ptr, config.n_layers, config.dim * (config.n_kv_heads * head_size));
        weights.wo = init_quantized_tensors(&ptr, config.n_layers, (config.n_heads * head_size) * config.dim);

        weights.w1 = init_quantized_tensors(&ptr, config.n_layers, config.dim * config.hidden_dim);
        weights.w2 = init_quantized_tensors(&ptr, config.n_layers, config.hidden_dim * config.dim);
        weights.w3 = init_quantized_tensors(&ptr, config.n_layers, config.dim * config.hidden_dim);

        if (shared_weights) {
            weights.wcls = weights.q_tokens.data();
        } else {
            weights.wcls_storage = init_quantized_tensors(&ptr, 1, config.dim * config.vocab_size);
            weights.wcls = weights.wcls_storage.data();
        }
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

            int i = 0;
#if defined(__ARM_NEON)
            float32x4_t max_vec = vdupq_n_f32(0.0f);
            for (; i <= group_size - 16; i += 16) {
                float32x4_t v0 = vabsq_f32(vld1q_f32(x + group * group_size + i));
                float32x4_t v1 = vabsq_f32(vld1q_f32(x + group * group_size + i + 4));
                float32x4_t v2 = vabsq_f32(vld1q_f32(x + group * group_size + i + 8));
                float32x4_t v3 = vabsq_f32(vld1q_f32(x + group * group_size + i + 12));

                max_vec = vmaxnmq_f32(max_vec, v0);
                max_vec = vmaxnmq_f32(max_vec, v1);
                max_vec = vmaxnmq_f32(max_vec, v2);
                max_vec = vmaxnmq_f32(max_vec, v3);
            }
            // Reduction for max
            wmax = vmaxnmvq_f32(max_vec);
#endif
            for (; i < group_size; i++) {
                if (const float val = std::fabs(x[group * group_size + i]); val > wmax) wmax = val;
            }

            const float scale = wmax / Q_MAX;
            qx->s[group] = scale;
            const float inv_scale = (scale != 0.0f) ? 1.0f / scale : 0.0f;

            i = 0;
#if defined(__ARM_NEON)
            float32x4_t inv_scale_vec = vdupq_n_f32(inv_scale);
            for (; i <= group_size - 16; i += 16) {
                float32x4_t v0 = vld1q_f32(x + group * group_size + i);
                float32x4_t v1 = vld1q_f32(x + group * group_size + i + 4);
                float32x4_t v2 = vld1q_f32(x + group * group_size + i + 8);
                float32x4_t v3 = vld1q_f32(x + group * group_size + i + 12);

                v0 = vmulq_f32(v0, inv_scale_vec);
                v1 = vmulq_f32(v1, inv_scale_vec);
                v2 = vmulq_f32(v2, inv_scale_vec);
                v3 = vmulq_f32(v3, inv_scale_vec);

                int32x4_t i0 = vcvtaq_s32_f32(v0);
                int32x4_t i1 = vcvtaq_s32_f32(v1);
                int32x4_t i2 = vcvtaq_s32_f32(v2);
                int32x4_t i3 = vcvtaq_s32_f32(v3);

                int16x8_t q01 = vcombine_s16(vqmovn_s32(i0), vqmovn_s32(i1));
                int16x8_t q23 = vcombine_s16(vqmovn_s32(i2), vqmovn_s32(i3));

                int8x16_t q_final = vcombine_s8(vqmovn_s16(q01), vqmovn_s16(q23));
                vst1q_s8(qx->q + group * group_size + i, q_final);
            }
#endif
            for (; i < group_size; i++) {
                const float quant_value = x[group * group_size + i] * inv_scale;
                const auto quantized = static_cast<int8_t>(std::round(quant_value));
                qx->q[group * group_size + i] = quantized;
            }
        }
    }

    void QuantizedTransformer::rmsnorm(float *o, const float *x, const float *weight, const int size) {
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

    void QuantizedTransformer::softmax(float *x, const int size) {
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

    void QuantizedTransformer::matmul(float *__restrict__ xout, const QuantizedTensor *__restrict__ x,
                                      const QuantizedTensor *__restrict__ w, const int n,
                                      const int d) const {
        int i;
#pragma omp parallel for private(i) schedule(static)
        for (i = 0; i < d; i += 4) {
            if (i + 3 < d) {
                float val0 = 0.0f;
                float val1 = 0.0f;
                float val2 = 0.0f;
                float val3 = 0.0f;

                const int in0 = i * n;
                const int in1 = (i + 1) * n;
                const int in2 = (i + 2) * n;
                const int in3 = (i + 3) * n;

#if defined(__ARM_NEON)
                for (int j = 0; j <= n - group_size; j += group_size) {
                    int32_t ival0 = 0;
                    int32_t ival1 = 0;
                    int32_t ival2 = 0;
                    int32_t ival3 = 0;

                    int k = 0;
                    int32x4_t sum0 = vdupq_n_s32(0);
                    int32x4_t sum1 = vdupq_n_s32(0);
                    int32x4_t sum2 = vdupq_n_s32(0);
                    int32x4_t sum3 = vdupq_n_s32(0);

                    for (; k <= group_size - 16; k += 16) {
                        const int8x16_t qx_vec = vld1q_s8(x->q + j + k);

                        const int8x16_t qw0 = vld1q_s8(w->q + in0 + j + k);
                        const int8x16_t qw1 = vld1q_s8(w->q + in1 + j + k);
                        const int8x16_t qw2 = vld1q_s8(w->q + in2 + j + k);
                        const int8x16_t qw3 = vld1q_s8(w->q + in3 + j + k);

#if defined(__ARM_FEATURE_DOTPROD)
                sum0 = vdotq_s32(sum0, qx_vec, qw0);
                sum1 = vdotq_s32(sum1, qx_vec, qw1);
                sum2 = vdotq_s32(sum2, qx_vec, qw2);
                sum3 = vdotq_s32(sum3, qx_vec, qw3);
#else
                int16x8_t ml0_lo = vmull_s8(vget_low_s8(qx_vec), vget_low_s8(qw0));
                int16x8_t ml0_hi = vmull_s8(vget_high_s8(qx_vec), vget_high_s8(qw0));
                sum0 = vaddw_s16(sum0, vget_low_s16(ml0_lo));
                sum0 = vaddw_s16(sum0, vget_high_s16(ml0_lo));
                sum0 = vaddw_s16(sum0, vget_low_s16(ml0_hi));
                sum0 = vaddw_s16(sum0, vget_high_s16(ml0_hi));

                int16x8_t ml1_lo = vmull_s8(vget_low_s8(qx_vec), vget_low_s8(qw1));
                int16x8_t ml1_hi = vmull_s8(vget_high_s8(qx_vec), vget_high_s8(qw1));
                sum1 = vaddw_s16(sum1, vget_low_s16(ml1_lo));
                sum1 = vaddw_s16(sum1, vget_high_s16(ml1_lo));
                sum1 = vaddw_s16(sum1, vget_low_s16(ml1_hi));
                sum1 = vaddw_s16(sum1, vget_high_s16(ml1_hi));

                int16x8_t ml2_lo = vmull_s8(vget_low_s8(qx_vec), vget_low_s8(qw2));
                int16x8_t ml2_hi = vmull_s8(vget_high_s8(qx_vec), vget_high_s8(qw2));
                sum2 = vaddw_s16(sum2, vget_low_s16(ml2_lo));
                sum2 = vaddw_s16(sum2, vget_high_s16(ml2_lo));
                sum2 = vaddw_s16(sum2, vget_low_s16(ml2_hi));
                sum2 = vaddw_s16(sum2, vget_high_s16(ml2_hi));

                int16x8_t ml3_lo = vmull_s8(vget_low_s8(qx_vec), vget_low_s8(qw3));
                int16x8_t ml3_hi = vmull_s8(vget_high_s8(qx_vec), vget_high_s8(qw3));
                sum3 = vaddw_s16(sum3, vget_low_s16(ml3_lo));
                sum3 = vaddw_s16(sum3, vget_high_s16(ml3_lo));
                sum3 = vaddw_s16(sum3, vget_low_s16(ml3_hi));
                sum3 = vaddw_s16(sum3, vget_high_s16(ml3_hi));
#endif
                    }
                ival0 = vaddvq_s32(sum0);
                ival1 = vaddvq_s32(sum1);
                ival2 = vaddvq_s32(sum2);
                ival3 = vaddvq_s32(sum3);

                for (; k < group_size; k++) {
                    int8_t qx = x->q[j + k];
                    ival0 += static_cast<int32_t>(qx) * static_cast<int32_t>(w->q[in0 + j + k]);
                    ival1 += static_cast<int32_t>(qx) * static_cast<int32_t>(w->q[in1 + j + k]);
                    ival2 += static_cast<int32_t>(qx) * static_cast<int32_t>(w->q[in2 + j + k]);
                    ival3 += static_cast<int32_t>(qx) * static_cast<int32_t>(w->q[in3 + j + k]);
                }

                float sx = x->s[j / group_size];
                val0 += static_cast<float>(ival0) * w->s[(in0 + j) / group_size] * sx;
                val1 += static_cast<float>(ival1) * w->s[(in1 + j) / group_size] * sx;
                val2 += static_cast<float>(ival2) * w->s[(in2 + j) / group_size] * sx;
                val3 += static_cast<float>(ival3) * w->s[(in3 + j) / group_size] * sx;
                }
#else
                for (int j = 0; j <= n - group_size; j += group_size) {
                    int32_t ival0 = 0, ival1 = 0, ival2 = 0, ival3 = 0;
                    for (int k = 0; k < group_size; k++) {
                        int8_t qx = x->q[j + k];
                        ival0 += static_cast<int32_t>(qx) * static_cast<int32_t>(w->q[in0 + j + k]);
                        ival1 += static_cast<int32_t>(qx) * static_cast<int32_t>(w->q[in1 + j + k]);
                        ival2 += static_cast<int32_t>(qx) * static_cast<int32_t>(w->q[in2 + j + k]);
                        ival3 += static_cast<int32_t>(qx) * static_cast<int32_t>(w->q[in3 + j + k]);
                    }
                    float sx = x->s[j / group_size];
                    val0 += static_cast<float>(ival0) * w->s[(in0 + j) / group_size] * sx;
                    val1 += static_cast<float>(ival1) * w->s[(in1 + j) / group_size] * sx;
                    val2 += static_cast<float>(ival2) * w->s[(in2 + j) / group_size] * sx;
                    val3 += static_cast<float>(ival3) * w->s[(in3 + j) / group_size] * sx;
                }
#endif
                xout[i] = val0;
                xout[i + 1] = val1;
                xout[i + 2] = val2;
                xout[i + 3] = val3;
            } else {
                for (int k = i; k < d; k++) {
                    float val = 0.0f;
                    const int in = k * n;

#if defined(__ARM_NEON)
                    for (int j = 0; j <= n - group_size; j += group_size) {
                        int32_t ival = 0;
                        int k_idx = 0;
                        int32x4_t sum_vec = vdupq_n_s32(0);

                        for (; k_idx <= group_size - 16; k_idx += 16) {
                            const int8x16_t qx_vec = vld1q_s8(x->q + j + k_idx);
                            const int8x16_t qw_vec = vld1q_s8(w->q + in + j + k_idx);

#if defined(__ARM_FEATURE_DOTPROD)
                    sum_vec = vdotq_s32(sum_vec, qx_vec, qw_vec);
#else
                    int16x8_t mul_low = vmull_s8(vget_low_s8(qx_vec), vget_low_s8(qw_vec));
                    int16x8_t mul_high = vmull_s8(vget_high_s8(qx_vec), vget_high_s8(qw_vec));
                    sum_vec = vaddw_s16(sum_vec, vget_low_s16(mul_low));
                    sum_vec = vaddw_s16(sum_vec, vget_high_s16(mul_low));
                    sum_vec = vaddw_s16(sum_vec, vget_low_s16(mul_high));
                    sum_vec = vaddw_s16(sum_vec, vget_high_s16(mul_high));
#endif
                        }
                    ival = vaddvq_s32(sum_vec);
                    for (; k_idx < group_size; k_idx++) {
                        ival += static_cast<int32_t>(x->q[j + k_idx]) * static_cast<int32_t>(w->q[in + j + k_idx]);
                    }
                    val += static_cast<float>(ival) * w->s[(in + j) / group_size] * x->s[j / group_size];
                    }
#else
                    for (int j = 0; j <= n - group_size; j += group_size) {
                        int32_t ival = 0;
                        for (int k_sub = 0; k_sub < group_size; k_sub++) {
                            ival += static_cast<int32_t>(x->q[j + k_sub]) * static_cast<int32_t>(w->q[in + j + k_sub]);
                        }
                        val += static_cast<float>(ival) * w->s[(in + j) / group_size] * x->s[j / group_size];
                    }
#endif
                    xout[k] = val;
                }
            }
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

        std::copy_n(w->token_embedding_table.data() + token * dim, dim, x);

        for (int l = 0; l < p->n_layers; l++) {
            rmsnorm(s->xb.data(), x, w->rms_att_weight + l * dim, dim);

            // Construct temp QuantizedTensor for xq
            QuantizedTensor xq_tensor{s->xq_q.data(), s->xq_s.data()};
            quantize(&xq_tensor, s->xb.data(), dim);

            matmul(s->q.data(), &xq_tensor, &w->wq[l], dim, dim);
            matmul(s->k.data(), &xq_tensor, &w->wk[l], dim, kv_dim);
            matmul(s->v.data(), &xq_tensor, &w->wv[l], dim, kv_dim);

            // RoPE
            const int rope_offset = pos * (head_size / 2);
            for (int i = 0; i < p->n_heads; i++) {
                int j = 0;
#if defined(__ARM_NEON)
                for (; j <= head_size - 4; j += 4) {
                    float32x4_t q_vec = vld1q_f32(s->q.data() + i * head_size + j);

                    float c0 = rope_cos[rope_offset + j / 2];
                    float c1 = rope_cos[rope_offset + j / 2 + 1];
                    float s0 = rope_sin[rope_offset + j / 2];
                    float s1 = rope_sin[rope_offset + j / 2 + 1];

                    float32x4_t c_vec = {c0, c0, c1, c1};
                    float32x4_t s_signed = {-s0, s0, -s1, s1};

                    float32x4_t term1 = vmulq_f32(q_vec, c_vec);
                    float32x4_t q_swap = vrev64q_f32(q_vec);
                    float32x4_t term2 = vmulq_f32(q_swap, s_signed);
                    float32x4_t res = vaddq_f32(term1, term2);
                    vst1q_f32(s->q.data() + i * head_size + j, res);

                    if (i < p->n_kv_heads) {
                        float32x4_t k_vec = vld1q_f32(s->k.data() + i * head_size + j);
                        float32x4_t k_term1 = vmulq_f32(k_vec, c_vec);
                        float32x4_t k_swap = vrev64q_f32(k_vec);
                        float32x4_t k_term2 = vmulq_f32(k_swap, s_signed);
                        float32x4_t k_res = vaddq_f32(k_term1, k_term2);
                        vst1q_f32(s->k.data() + i * head_size + j, k_res);
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
            std::copy_n(s->k.data(), kv_dim, key_cache_row);
            std::copy_n(s->v.data(), kv_dim, value_cache_row);

            int h;
#pragma omp parallel for private(h)
            for (h = 0; h < p->n_heads; h++) {
                const float *q = s->q.data() + h * head_size;
                float *att = s->att.data() + h * p->seq_len;
                for (int t = 0; t <= pos; t++) {
                    const float *k = s->key_cache.data() + loff + t * kv_dim + (h / kv_mul) * head_size;

                    float score = 0.0f;
                    int i = 0;
#if defined(__ARM_NEON)
                    float32x4_t sum_vec = vdupq_n_f32(0.0f);
                    for (; i <= head_size - 4; i += 4) {
                        float32x4_t q_vec = vld1q_f32(q + i);
                        float32x4_t k_vec = vld1q_f32(k + i); // Corrected pointer from k_ptr to k
                        sum_vec = vmlaq_f32(sum_vec, q_vec, k_vec);
                    }
                    score = vaddvq_f32(sum_vec);
#endif
                    for (; i < head_size; i++) {
                        score += q[i] * k[i];
                    }
                    score /= std::sqrt(static_cast<float>(head_size));
                    att[t] = score;
                }

                softmax(att, pos + 1);

                float *xb = s->xb.data() + h * head_size;
                std::fill_n(xb, head_size, 0.0f);

                for (int t = 0; t <= pos; t++) {
                    const float *v = s->value_cache.data() + loff + t * kv_dim + (h / kv_mul) * head_size;
                    const float a = att[t];
                    int i = 0;
#if defined(__ARM_NEON)
                    float32x4_t a_vec = vdupq_n_f32(a);
                    for (; i <= head_size - 4; i += 4) {
                        float32x4_t xb_vec = vld1q_f32(xb + i);
                        float32x4_t v_vec = vld1q_f32(v + i); // Corrected pointer from v_ptr to v
                        xb_vec = vmlaq_f32(xb_vec, a_vec, v_vec);
                        vst1q_f32(xb + i, xb_vec);
                    }
#endif
                    for (; i < head_size; i++) {
                        xb[i] += a * v[i];
                    }
                }
            }

            quantize(&xq_tensor, s->xb.data(), dim);
            matmul(s->xb2.data(), &xq_tensor, &w->wo[l], dim, dim);

            for (int i = 0; i < dim; i++) {
                x[i] += s->xb2[i];
            }

            rmsnorm(s->xb.data(), x, w->rms_ffn_weight + l * dim, dim);

            quantize(&xq_tensor, s->xb.data(), dim);
            matmul(s->hb.data(), &xq_tensor, &w->w1[l], dim, hidden_dim);
            matmul(s->hb2.data(), &xq_tensor, &w->w3[l], dim, hidden_dim);

            for (int i = 0; i < hidden_dim; i++) {
                float val = s->hb[i];
                val *= (1.0f / (1.0f + std::exp(-val)));
                val *= s->hb2[i];
                s->hb[i] = val;
            }

            QuantizedTensor hq_tensor{s->hq_q.data(), s->hq_s.data()};
            quantize(&hq_tensor, s->hb.data(), hidden_dim);
            matmul(s->xb.data(), &hq_tensor, &w->w2[l], hidden_dim, dim);

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