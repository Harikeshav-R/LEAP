#include "FloatTransformer.h"
#include "../kernel/leap_protocol.h"
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

#if defined(__x86_64__) || defined(_M_X64)
#include <immintrin.h>
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
        state.k.resize(kv_dim);
        state.v.resize(kv_dim);
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
            const float32x4_t val = vld1q_f32(x + j);

            sum_vec = vmlaq_f32(sum_vec, val, val);
        }

        ss = vaddvq_f32(sum_vec);

        for (; j < size; j++) {
            ss += x[j] * x[j];
        }

#elif defined(__AVX2__)

        __m256 sum_vec = _mm256_setzero_ps();

        int j = 0;

        for (; j <= size - 8; j += 8) {
            __m256 val = _mm256_loadu_ps(x + j);

            sum_vec = _mm256_fmadd_ps(val, val, sum_vec);
        }

        // Horizontal sum

        // 1. Extract high 128 bits

        __m128 sum_high = _mm256_extractf128_ps(sum_vec, 1);

        __m128 sum_low = _mm256_castps256_ps128(sum_vec);

        __m128 sum128 = _mm_add_ps(sum_low, sum_high);

        // 2. Horizontal add within 128 bits

        sum128 = _mm_hadd_ps(sum128, sum128);

        sum128 = _mm_hadd_ps(sum128, sum128);

        ss = _mm_cvtss_f32(sum128);


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

#if defined(__ARM_NEON)
        const float32x4_t ss_v = vdupq_n_f32(ss);
        float32x4_t rsqrt_est = vrsqrteq_f32(ss_v);
        const float32x4_t rsqrt_step = vrsqrtsq_f32(ss_v, vmulq_f32(rsqrt_est, rsqrt_est));
        rsqrt_est = vmulq_f32(rsqrt_est, rsqrt_step);
        ss = vgetq_lane_f32(rsqrt_est, 0);
#elif defined(__AVX2__)
        __m128 ss_v = _mm_set_ss(ss);
        __m128 rsqrt_est = _mm_rsqrt_ss(ss_v);
        __m128 three = _mm_set_ss(3.0f);
        __m128 half = _mm_set_ss(0.5f);
        __m128 est_sq = _mm_mul_ss(rsqrt_est, rsqrt_est);
        __m128 term = _mm_sub_ss(three, _mm_mul_ss(ss_v, est_sq));
        rsqrt_est = _mm_mul_ss(_mm_mul_ss(half, rsqrt_est), term);
        ss = _mm_cvtss_f32(rsqrt_est);
#else
        ss = 1.0f / std::sqrt(ss);
#endif

#if defined(__ARM_NEON)

        j = 0;

        const float32x4_t ss_vec = vdupq_n_f32(ss);

        for (; j <= size - 4; j += 4) {
            const float32x4_t w_val = vld1q_f32(weight + j);

            const float32x4_t x_val = vld1q_f32(x + j);

            const float32x4_t res = vmulq_f32(w_val, vmulq_f32(ss_vec, x_val));

            vst1q_f32(o + j, res);
        }

        for (; j < size; j++) {
            o[j] = weight[j] * (ss * x[j]);
        }

#elif defined(__AVX2__)

        j = 0;

        __m256 ss_vec = _mm256_set1_ps(ss);

        for (; j <= size - 8; j += 8) {
            __m256 w_val = _mm256_loadu_ps(weight + j);

            __m256 x_val = _mm256_loadu_ps(x + j);

            __m256 res = _mm256_mul_ps(w_val, _mm256_mul_ps(ss_vec, x_val));

            _mm256_storeu_ps(o + j, res);
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
        for (i = 0; i < d; i += 8) {
            if (i + 7 < d) {
                float val0 = 0.0f;
                float val1 = 0.0f;
                float val2 = 0.0f;
                float val3 = 0.0f;
                float val4 = 0.0f;
                float val5 = 0.0f;
                float val6 = 0.0f;
                float val7 = 0.0f;

                int j = 0;
                const int offset0 = i * n;
                const int offset1 = (i + 1) * n;
                const int offset2 = (i + 2) * n;
                const int offset3 = (i + 3) * n;
                const int offset4 = (i + 4) * n;
                const int offset5 = (i + 5) * n;
                const int offset6 = (i + 6) * n;
                const int offset7 = (i + 7) * n;

#if defined(__ARM_NEON)
                float32x4_t sum0 = vdupq_n_f32(0.0f);
                float32x4_t sum1 = vdupq_n_f32(0.0f);
                float32x4_t sum2 = vdupq_n_f32(0.0f);
                float32x4_t sum3 = vdupq_n_f32(0.0f);
                float32x4_t sum4 = vdupq_n_f32(0.0f);
                float32x4_t sum5 = vdupq_n_f32(0.0f);
                float32x4_t sum6 = vdupq_n_f32(0.0f);
                float32x4_t sum7 = vdupq_n_f32(0.0f);

                for (; j <= n - 4; j += 4) {
                    float32x4_t x_vec = vld1q_f32(x + j);

                    sum0 = vmlaq_f32(sum0, vld1q_f32(w + offset0 + j), x_vec);
                    sum1 = vmlaq_f32(sum1, vld1q_f32(w + offset1 + j), x_vec);
                    sum2 = vmlaq_f32(sum2, vld1q_f32(w + offset2 + j), x_vec);
                    sum3 = vmlaq_f32(sum3, vld1q_f32(w + offset3 + j), x_vec);
                    sum4 = vmlaq_f32(sum4, vld1q_f32(w + offset4 + j), x_vec);
                    sum5 = vmlaq_f32(sum5, vld1q_f32(w + offset5 + j), x_vec);
                    sum6 = vmlaq_f32(sum6, vld1q_f32(w + offset6 + j), x_vec);
                    sum7 = vmlaq_f32(sum7, vld1q_f32(w + offset7 + j), x_vec);
                }
                val0 = vaddvq_f32(sum0);
                val1 = vaddvq_f32(sum1);
                val2 = vaddvq_f32(sum2);
                val3 = vaddvq_f32(sum3);
                val4 = vaddvq_f32(sum4);
                val5 = vaddvq_f32(sum5);
                val6 = vaddvq_f32(sum6);
                val7 = vaddvq_f32(sum7);
#elif defined(__AVX2__)
                __m256 sum0 = _mm256_setzero_ps(); __m256 sum1 = _mm256_setzero_ps();
                __m256 sum2 = _mm256_setzero_ps(); __m256 sum3 = _mm256_setzero_ps();
                __m256 sum4 = _mm256_setzero_ps(); __m256 sum5 = _mm256_setzero_ps();
                __m256 sum6 = _mm256_setzero_ps(); __m256 sum7 = _mm256_setzero_ps();

                for (; j <= n - 8; j += 8) {
                    __m256 x_vec = _mm256_loadu_ps(x + j);

                    sum0 = _mm256_fmadd_ps(_mm256_loadu_ps(w + offset0 + j), x_vec, sum0);
                    sum1 = _mm256_fmadd_ps(_mm256_loadu_ps(w + offset1 + j), x_vec, sum1);
                    sum2 = _mm256_fmadd_ps(_mm256_loadu_ps(w + offset2 + j), x_vec, sum2);
                    sum3 = _mm256_fmadd_ps(_mm256_loadu_ps(w + offset3 + j), x_vec, sum3);
                    sum4 = _mm256_fmadd_ps(_mm256_loadu_ps(w + offset4 + j), x_vec, sum4);
                    sum5 = _mm256_fmadd_ps(_mm256_loadu_ps(w + offset5 + j), x_vec, sum5);
                    sum6 = _mm256_fmadd_ps(_mm256_loadu_ps(w + offset6 + j), x_vec, sum6);
                    sum7 = _mm256_fmadd_ps(_mm256_loadu_ps(w + offset7 + j), x_vec, sum7);
                }

                auto hsum256_ps = [](__m256 v) -> float {
                    __m128 v_low = _mm256_castps256_ps128(v);
                    __m128 v_high = _mm256_extractf128_ps(v, 1);
                    v_low = _mm_add_ps(v_low, v_high);
                    v_low = _mm_hadd_ps(v_low, v_low);
                    v_low = _mm_hadd_ps(v_low, v_low);
                    return _mm_cvtss_f32(v_low);
                };

                val0 = hsum256_ps(sum0); val1 = hsum256_ps(sum1);
                val2 = hsum256_ps(sum2); val3 = hsum256_ps(sum3);
                val4 = hsum256_ps(sum4); val5 = hsum256_ps(sum5);
                val6 = hsum256_ps(sum6); val7 = hsum256_ps(sum7);
#endif
                for (; j < n; j++) {
                    float xv = x[j];
                    val0 += w[offset0 + j] * xv;
                    val1 += w[offset1 + j] * xv;
                    val2 += w[offset2 + j] * xv;
                    val3 += w[offset3 + j] * xv;
                    val4 += w[offset4 + j] * xv;
                    val5 += w[offset5 + j] * xv;
                    val6 += w[offset6 + j] * xv;
                    val7 += w[offset7 + j] * xv;
                }
                xout[i] = val0;
                xout[i + 1] = val1;
                xout[i + 2] = val2;
                xout[i + 3] = val3;
                xout[i + 4] = val4;
                xout[i + 5] = val5;
                xout[i + 6] = val6;
                xout[i + 7] = val7;
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
#elif defined(__AVX2__)
                    __m256 sum_vec = _mm256_setzero_ps();
                    for (; j <= n - 8; j += 8) {
                        __m256 w_vec = _mm256_loadu_ps(w + offset + j);
                        __m256 x_vec = _mm256_loadu_ps(x + j);
                        sum_vec = _mm256_fmadd_ps(w_vec, x_vec, sum_vec);
                    }
                    __m128 sum_high = _mm256_extractf128_ps(sum_vec, 1);
                    __m128 sum_low = _mm256_castps256_ps128(sum_vec);
                    __m128 sum128 = _mm_add_ps(sum_low, sum_high);
                    sum128 = _mm_hadd_ps(sum128, sum128);
                    sum128 = _mm_hadd_ps(sum128, sum128);
                    val = _mm_cvtss_f32(sum128);
#endif
                    for (; j < n; j++) {
                        val += w[offset + j] * x[j];
                    }
                    xout[k] = val;
                }
            }
        }
    }

    void FloatTransformer::run_layer(int l, int pos, float *x) {
        const Config *p = &config;
        const FloatTransformerWeights *w = &weights;
        FloatRunState *s = &state;
        const int dim = p->dim;
        const int kv_dim = (p->dim * p->n_kv_heads) / p->n_heads;
        const int kv_mul = p->n_heads / p->n_kv_heads;
        const int hidden_dim = p->hidden_dim;
        const int head_size = dim / p->n_heads;

        rmsnorm(s->xb.data(), x, w->rms_att_weight + l * dim, dim);

        const int loff = l * p->seq_len * kv_dim;
        float *k = s->key_cache.data() + loff + pos * kv_dim;
        float *v = s->value_cache.data() + loff + pos * kv_dim;

        matmul(s->q.data(), s->xb.data(), w->wq + l * dim * dim, dim, dim);
        matmul(k, s->xb.data(), w->wk + l * dim * kv_dim, dim, kv_dim);
        matmul(v, s->xb.data(), w->wv + l * dim * kv_dim, dim, kv_dim);

        // RoPE
        const int rope_offset = pos * (head_size / 2);
#pragma omp parallel for
        for (int i = 0; i < p->n_heads; i++) {
            int j = 0;
#if defined(__ARM_NEON)
            for (; j <= head_size - 4; j += 4) {
                // Load 4 floats: [q0, q1, q2, q3] -> 2 pairs (q0,q1), (q2,q3)
                float32x4_t q_vec = vld1q_f32(s->q.data() + i * head_size + j);

                float c0 = rope_cos[rope_offset + j / 2];
                float c1 = rope_cos[rope_offset + j / 2 + 1];
                float s0 = rope_sin[rope_offset + j / 2];
                float s1 = rope_sin[rope_offset + j / 2 + 1];

                // Construct vectors: [c0, c0, c1, c1]
                float32x4_t c_vec = {c0, c0, c1, c1};
                float32x4_t s_vec = {s0, s0, s1, s1};

                // Step 1: Compute q * c
                float32x4_t term1 = vmulq_f32(q_vec, c_vec);

                // Step 2: Swap q to get [q1, q0, q3, q2]
                float32x4_t q_swap = vrev64q_f32(q_vec);

                // Step 3: Compute swapped_q * s
                // Multiply q_swap by [-s0, s0, -s1, s1].
                float32x4_t s_signed = {-s0, s0, -s1, s1};
                float32x4_t term2 = vmulq_f32(q_swap, s_signed);

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
#elif defined(__AVX2__)
            for (; j <= head_size - 8; j += 8) {
                __m256 q_vec = _mm256_loadu_ps(s->q.data() + i * head_size + j);

                __m128 c_small = _mm_loadu_ps(&rope_cos[rope_offset + j / 2]);
                __m128 s_small = _mm_loadu_ps(&rope_sin[rope_offset + j / 2]);

                __m128 c_lo = _mm_unpacklo_ps(c_small, c_small); // c0, c0, c1, c1
                __m128 c_hi = _mm_unpackhi_ps(c_small, c_small); // c2, c2, c3, c3
                __m256 c_vec = _mm256_insertf128_ps(_mm256_castps128_ps256(c_lo), c_hi, 1);

                __m128 s_lo = _mm_unpacklo_ps(s_small, s_small);
                __m128 s_hi = _mm_unpackhi_ps(s_small, s_small);
                __m256 s_vec = _mm256_insertf128_ps(_mm256_castps128_ps256(s_lo), s_hi, 1);

                __m256 q_swap = _mm256_permute_ps(q_vec, _MM_SHUFFLE(2, 3, 0, 1));
                __m256 term1 = _mm256_mul_ps(q_vec, c_vec);
                __m256 term2 = _mm256_mul_ps(q_swap, s_vec);
                __m256 res = _mm256_addsub_ps(term1, term2);

                _mm256_storeu_ps(s->q.data() + i * head_size + j, res);

                if (i < p->n_kv_heads) {
                    __m256 k_vec = _mm256_loadu_ps(k + i * head_size + j);
                    __m256 k_swap = _mm256_permute_ps(k_vec, _MM_SHUFFLE(2, 3, 0, 1));
                    __m256 k_t1 = _mm256_mul_ps(k_vec, c_vec);
                    __m256 k_t2 = _mm256_mul_ps(k_swap, s_vec);
                    __m256 k_res = _mm256_addsub_ps(k_t1, k_t2);
                    _mm256_storeu_ps(k + i * head_size + j, k_res);
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
            int t = 0;

            // Unrolled loop for t
            for (; t <= pos - 4; t += 4) {
                const float *k0 = s->key_cache.data() + loff + t * kv_dim + (h / kv_mul) * head_size;
                const float *k1 = s->key_cache.data() + loff + (t + 1) * kv_dim + (h / kv_mul) * head_size;
                const float *k2 = s->key_cache.data() + loff + (t + 2) * kv_dim + (h / kv_mul) * head_size;
                const float *k3 = s->key_cache.data() + loff + (t + 3) * kv_dim + (h / kv_mul) * head_size;

                float s0 = 0.0f;
                float s1 = 0.0f;
                float s2 = 0.0f;
                float s3 = 0.0f;

                int i = 0;
#if defined(__AVX2__)
                __m256 sum0 = _mm256_setzero_ps();
                __m256 sum1 = _mm256_setzero_ps();
                __m256 sum2 = _mm256_setzero_ps();
                __m256 sum3 = _mm256_setzero_ps();

                for (; i <= head_size - 8; i += 8) {
                    __m256 q_vec = _mm256_loadu_ps(q + i);
                    __m256 k0_vec = _mm256_loadu_ps(k0 + i);
                    __m256 k1_vec = _mm256_loadu_ps(k1 + i);
                    __m256 k2_vec = _mm256_loadu_ps(k2 + i);
                    __m256 k3_vec = _mm256_loadu_ps(k3 + i);

                    sum0 = _mm256_fmadd_ps(q_vec, k0_vec, sum0);
                    sum1 = _mm256_fmadd_ps(q_vec, k1_vec, sum1);
                    sum2 = _mm256_fmadd_ps(q_vec, k2_vec, sum2);
                    sum3 = _mm256_fmadd_ps(q_vec, k3_vec, sum3);
                }

                auto hsum = [](__m256 v) {
                    __m128 lo = _mm256_castps256_ps128(v);
                    __m128 hi = _mm256_extractf128_ps(v, 1);
                    lo = _mm_add_ps(lo, hi);
                    lo = _mm_hadd_ps(lo, lo);
                    return _mm_cvtss_f32(_mm_hadd_ps(lo, lo));
                };
                s0 = hsum(sum0);
                s1 = hsum(sum1);
                s2 = hsum(sum2);
                s3 = hsum(sum3);
#elif defined(__ARM_NEON)
                float32x4_t sum0_v = vdupq_n_f32(0.0f);
                float32x4_t sum1_v = vdupq_n_f32(0.0f);
                float32x4_t sum2_v = vdupq_n_f32(0.0f);
                float32x4_t sum3_v = vdupq_n_f32(0.0f);

                for (; i <= head_size - 4; i += 4) {
                    float32x4_t q_vec = vld1q_f32(q + i);
                    float32x4_t k0_vec = vld1q_f32(k0 + i);
                    float32x4_t k1_vec = vld1q_f32(k1 + i);
                    float32x4_t k2_vec = vld1q_f32(k2 + i);
                    float32x4_t k3_vec = vld1q_f32(k3 + i);

                    sum0_v = vmlaq_f32(sum0_v, q_vec, k0_vec);
                    sum1_v = vmlaq_f32(sum1_v, q_vec, k1_vec);
                    sum2_v = vmlaq_f32(sum2_v, q_vec, k2_vec);
                    sum3_v = vmlaq_f32(sum3_v, q_vec, k3_vec);
                }
                s0 = vaddvq_f32(sum0_v);
                s1 = vaddvq_f32(sum1_v);
                s2 = vaddvq_f32(sum2_v);
                s3 = vaddvq_f32(sum3_v);
#endif
                for (; i < head_size; i++) {
                    float qv = q[i];
                    s0 += qv * k0[i];
                    s1 += qv * k1[i];
                    s2 += qv * k2[i];
                    s3 += qv * k3[i];
                }

                float scale = 1.0f / std::sqrt(static_cast<float>(head_size));
                att[t] = s0 * scale;
                att[t + 1] = s1 * scale;
                att[t + 2] = s2 * scale;
                att[t + 3] = s3 * scale;
            }

            // Tail loop
            for (; t <= pos; t++) {
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
#elif defined(__AVX2__)
                __m256 sum_vec = _mm256_setzero_ps();
                for (; i <= head_size - 8; i += 8) {
                    __m256 q_vec = _mm256_loadu_ps(q + i);
                    __m256 k_vec = _mm256_loadu_ps(k_ptr + i);
                    sum_vec = _mm256_fmadd_ps(q_vec, k_vec, sum_vec);
                }
                // Horizontal sum
                auto hsum = [](__m256 v) {
                    __m128 lo = _mm256_castps256_ps128(v);
                    __m128 hi = _mm256_extractf128_ps(v, 1);
                    lo = _mm_add_ps(lo, hi);
                    lo = _mm_hadd_ps(lo, lo);
                    return _mm_cvtss_f32(_mm_hadd_ps(lo, lo));
                };
                score = hsum(sum_vec);
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
#elif defined(__AVX2__)
                __m256 a_vec = _mm256_set1_ps(a);
                for (; i <= head_size - 8; i += 8) {
                    __m256 xb_vec = _mm256_loadu_ps(xb + i);
                    __m256 v_vec = _mm256_loadu_ps(v_ptr + i);
                    xb_vec = _mm256_fmadd_ps(a_vec, v_vec, xb_vec);
                    _mm256_storeu_ps(xb + i, xb_vec);
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

#pragma omp parallel for simd
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

    float *FloatTransformer::forward(const int token, const int pos, const int flags) {
        const Config *p = &config;
        const FloatTransformerWeights *w = &weights;
        FloatRunState *s = &state;
        float *x = s->x.data();
        const int dim = p->dim;

        const float *content_row = w->token_embedding_table + token * dim;
        std::copy_n(content_row, dim, x);

        int start_layer = 0;
        int end_layer = p->n_layers;

        if (dist_config.mode == DistributedMode::Master) {
            end_layer = dist_config.split_layer;
        }

        for (int l = start_layer; l < end_layer; l++) {
            run_layer(l, pos, x);
        }

        if (dist_config.mode == DistributedMode::Master) {
            if (!dist_config.transport) throw std::runtime_error("Transport not set for master");

            // Resize reusable buffer if needed
            size_t packet_size = sizeof(PacketHeader) + dim * sizeof(float);
            if (transfer_buffer.size() < packet_size) transfer_buffer.resize(packet_size);

            PacketHeader header{pos, flags};
            
            std::memcpy(transfer_buffer.data(), &header, sizeof(PacketHeader));
            std::memcpy(transfer_buffer.data() + sizeof(PacketHeader), x, dim * sizeof(float));

            // Master sends to Next (Worker 1)
            dist_config.transport->send_next(transfer_buffer.data(), packet_size);

            if (flags == FLAG_NEED_REPLY) {
                // Ring Synchronization (Stop-and-Wait):
                // Only wait for reply if we actually asked for one.
                // For NO_REPLY (Prompt Phase), we Pipeline (Fire-and-Forget).
                
                dist_config.transport->recv_prev(transfer_buffer.data(), packet_size);
                
                // Extract the data (skip header)
                std::memcpy(x, transfer_buffer.data() + sizeof(PacketHeader), dim * sizeof(float));
            } else {
                return nullptr;
            }
        }

        rmsnorm(x, x, w->rms_final_weight, dim);
        matmul(s->logits.data(), x, w->wcls, dim, p->vocab_size);
        return s->logits.data();
    }

    void FloatTransformer::worker_loop() {
        if (!dist_config.transport) throw std::runtime_error("Transport not set for worker");

        float *x = state.x.data();
        const int dim = config.dim;
        PacketHeader header{};

        const int start_layer = dist_config.split_layer;
        
        size_t packet_size = sizeof(PacketHeader) + dim * sizeof(float);
        if (transfer_buffer.size() < packet_size) transfer_buffer.resize(packet_size);

        std::cout << "Worker started. Processing layers " << start_layer << " to " << dist_config.end_layer - 1 << std::endl;

        while (true) {
            try {
                // Receive header + data from Prev
                dist_config.transport->recv_prev(transfer_buffer.data(), packet_size);

                std::memcpy(&header, transfer_buffer.data(), sizeof(PacketHeader));
                std::memcpy(x, transfer_buffer.data() + sizeof(PacketHeader), dim * sizeof(float));

                // Process layers
                for (int l = start_layer; l < dist_config.end_layer; l++) {
                    run_layer(l, header.pos, x);
                }

                // If Tail and NO_REPLY, Drop the packet (End of Pipeline)
                // If Not Tail, Forward to Next
                // If Tail and NEED_REPLY, Forward to Next (Master)
                
                if (!dist_config.is_tail || header.flags == FLAG_NEED_REPLY) {
                    std::memcpy(transfer_buffer.data() + sizeof(PacketHeader), x, dim * sizeof(float));
                    dist_config.transport->send_next(transfer_buffer.data(), packet_size);
                }

            } catch (const std::exception &e) {
                std::cerr << "Worker loop error: " << e.what() << std::endl;
                break;
            }
        }
    }

    void FloatTransformer::memory_map_weights(const float *ptr, const int shared_weights) {
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