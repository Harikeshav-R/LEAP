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
            float32x4_t val = vld1q_f32(x + j);

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
#elif defined(__AVX2__)
                __m256 sum0 = _mm256_setzero_ps();
                __m256 sum1 = _mm256_setzero_ps();
                __m256 sum2 = _mm256_setzero_ps();
                __m256 sum3 = _mm256_setzero_ps();

                for (; j <= n - 8; j += 8) {
                    __m256 x_vec = _mm256_loadu_ps(x + j);

                    __m256 w0 = _mm256_loadu_ps(w + offset0 + j);
                    __m256 w1 = _mm256_loadu_ps(w + offset1 + j);
                    __m256 w2 = _mm256_loadu_ps(w + offset2 + j);
                    __m256 w3 = _mm256_loadu_ps(w + offset3 + j);

                    sum0 = _mm256_fmadd_ps(w0, x_vec, sum0);
                    sum1 = _mm256_fmadd_ps(w1, x_vec, sum1);
                    sum2 = _mm256_fmadd_ps(w2, x_vec, sum2);
                    sum3 = _mm256_fmadd_ps(w3, x_vec, sum3);
                }

                // Horizontal reduction helper
                auto hsum256_ps = [](__m256 v) -> float {
                    __m128 v_low = _mm256_castps256_ps128(v);
                    __m128 v_high = _mm256_extractf128_ps(v, 1);
                    v_low = _mm_add_ps(v_low, v_high);
                    v_low = _mm_hadd_ps(v_low, v_low);
                    v_low = _mm_hadd_ps(v_low, v_low);
                    return _mm_cvtss_f32(v_low);
                };

                val0 = hsum256_ps(sum0);
                val1 = hsum256_ps(sum1);
                val2 = hsum256_ps(sum2);
                val3 = hsum256_ps(sum3);
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
                        __builtin_prefetch(w + offset + j + 32, 0, 0);
                        __builtin_prefetch(x + j + 32, 0, 0);
                        float32x4_t w_vec = vld1q_f32(w + offset + j);
                        float32x4_t x_vec = vld1q_f32(x + j);
                        sum_vec = vmlaq_f32(sum_vec, w_vec, x_vec);
                    }
                    val = vaddvq_f32(sum_vec);
#elif defined(__AVX2__)
                    __m256 sum_vec = _mm256_setzero_ps();
                    for (; j <= n - 8; j += 8) {
                        _mm_prefetch(reinterpret_cast<const char *>(w + offset + j + 32), _MM_HINT_T0);
                        _mm_prefetch(reinterpret_cast<const char *>(x + j + 32), _MM_HINT_T0);
                        __m256 w_vec = _mm256_loadu_ps(w + offset + j);
                        __m256 x_vec = _mm256_loadu_ps(x + j);
                        sum_vec = _mm256_fmadd_ps(w_vec, x_vec, sum_vec);
                    }
                    // Horizontal sum
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
#elif defined(__AVX2__)
                for (; j <= head_size - 8; j += 8) {
                    // Load 8 q values
                    __m256 q_vec = _mm256_loadu_ps(s->q.data() + i * head_size + j);

                    // q_vec = [q0, q1, q2, q3, q4, q5, q6, q7]

                    // Need cos/sin:
                    // rope_cos is packed as [c0, c1, c2, c3] for pairs (0,1), (2,3), (4,5), (6,7)
                    // We need [c0, c0, c1, c1, c2, c2, c3, c3]

                    // Load 4 floats from rope_cos/sin
                    // rope_cos is float*
                    // We load pairs. The index calculation `rope_offset + j/2` gives the start.
                    // j=0 -> index 0. We need 4 values: c0, c1, c2, c3.
                    // This corresponds to rope_cos[offset + 0, 1, 2, 3]

                    __m128 c_small = _mm_loadu_ps(&rope_cos[rope_offset + j / 2]);
                    __m128 s_small = _mm_loadu_ps(&rope_sin[rope_offset + j / 2]);

                    // Duplicate each element: [c0, c1, c2, c3] -> [c0, c0, c1, c1, c2, c2, c3, c3]
                    // _mm256_cvtepi32_ps(_mm256_cvttps_epi32(...)) is slow.
                    // Use unpack/shuffle.

                    // unpacklo: [c0, c0, c1, c1] (if interacting with self)
                    __m128 c_lo = _mm_unpacklo_ps(c_small, c_small); // c0, c0, c1, c1
                    __m128 c_hi = _mm_unpackhi_ps(c_small, c_small); // c2, c2, c3, c3

                    // Combine into m256
                    __m256 c_vec = _mm256_insertf128_ps(_mm256_castps128_ps256(c_lo), c_hi, 1);

                    __m128 s_lo = _mm_unpacklo_ps(s_small, s_small);
                    __m128 s_hi = _mm_unpackhi_ps(s_small, s_small);
                    __m256 s_vec = _mm256_insertf128_ps(_mm256_castps128_ps256(s_lo), s_hi, 1);

                    // Create s_signed: [-s0, s0, -s1, s1, ...]
                    // XOR with sign bit mask 0x80000000 for even indices?
                    // Or just multiply.
                    // Standard formula:
                    // out[0] = q0*c0 - q1*s0
                    // out[1] = q0*s0 + q1*c0

                    // Swap q: [q1, q0, q3, q2, ...]
                    __m256 q_swap = _mm256_permute_ps(q_vec, _MM_SHUFFLE(2, 3, 0, 1));

                    // Term 1: q * c
                    __m256 term1 = _mm256_mul_ps(q_vec, c_vec);

                    // Term 2: q_swap * s_vec
                    __m256 term2 = _mm256_mul_ps(q_swap, s_vec);

                    // We need subtract for even indices, add for odd.
                    // q0*c0 - q1*s0
                    // q1*c0 + q0*s0 (Note: q_swap has q0 at odd index)
                    // So we want: term1 - term2 for even, term1 + term2 for odd.

                    // AVX2 has addsub (subtract even, add odd).
                    // _mm256_addsub_ps(a, b) = [a0-b0, a1+b1, ...]
                    // Wait, standard definition:
                    // "Subtracts odd positions, Adds even positions"?
                    // Intel guide: "dst[0] = a[0] - b[0]", "dst[1] = a[1] + b[1]".
                    // So yes, it subtracts for even index 0, adds for odd index 1.

                    __m256 res = _mm256_addsub_ps(term1, term2);
                    // But wait.
                    // even: q0*c0 - q1*s0. Matches (term1[0] - term2[0])
                    // odd:  q1*c0 + q0*s0. Matches (term1[1] + term2[1])
                    // BUT term2[1] is q0*s0?
                    // q_swap at 1 is q0. s_vec at 1 is s0. Yes.

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

                    // Reduce
                    // (Reuse hsum lambda if available, or manual)
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
                    __m128 sum_high = _mm256_extractf128_ps(sum_vec, 1);
                    __m128 sum_low = _mm256_castps256_ps128(sum_vec);
                    __m128 sum128 = _mm_add_ps(sum_low, sum_high);
                    sum128 = _mm_hadd_ps(sum128, sum128);
                    sum128 = _mm_hadd_ps(sum128, sum128);
                    score = _mm_cvtss_f32(sum128);
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

#pragma omp parallel for
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