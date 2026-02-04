#include "QuantizedTransformer.h"
#include "../kernel/leap_protocol.h"
#include <cmath>
#include <unistd.h>
#include <sys/mman.h>
#include <stdexcept>
#include <iostream>
#include <vector>
#include <algorithm>
#include <cstring>

#if defined(__ARM_NEON)
#include <arm_neon.h>
#endif

#if defined(__x86_64__) || defined(_M_X64)
#include <immintrin.h>
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
#if defined(__ARM_NEON)
        int i = 0;
        for (; i <= n - 16; i += 16) {
            // Load 16 int8 weights
            const int8x16_t q_raw = vld1q_s8(qx->q + i);

            // Expand to 16-bit
            const int16x8_t q_low = vmovl_s8(vget_low_s8(q_raw));
            const int16x8_t q_high = vmovl_s8(vget_high_s8(q_raw));

            // Expand to 32-bit float
            const float32x4_t f0 = vcvtq_f32_s32(vmovl_s16(vget_low_s16(q_low)));
            const float32x4_t f1 = vcvtq_f32_s32(vmovl_s16(vget_high_s16(q_low)));
            const float32x4_t f2 = vcvtq_f32_s32(vmovl_s16(vget_low_s16(q_high)));
            const float32x4_t f3 = vcvtq_f32_s32(vmovl_s16(vget_high_s16(q_high)));

            // Load scales. Group size is typically >= 32, so 16 elements share the same scale?
            // Wait, group_size can be 32, 64, 128.
            // If i is aligned to group_size, we load scale once.
            // But strict alignment isn't guaranteed inside the loop structure relative to group_size boundaries 
            // without more complex logic. 
            // However, dequantize is usually called on full blocks.
            // Let's assume standard group sizes and handle the general case by loading scale broadcast.

            // Optimization: If group_size >> 16, we can hoist scale loading.
            // For safety, we load scale for each block.

            const float s_val0 = qx->s[i / group_size];

            if (const float s_val1 = qx->s[(i + 15) / group_size]; s_val0 == s_val1) {
                const float32x4_t s_vec = vdupq_n_f32(s_val0);
                vst1q_f32(x + i, vmulq_f32(f0, s_vec));
                vst1q_f32(x + i + 4, vmulq_f32(f1, s_vec));
                vst1q_f32(x + i + 8, vmulq_f32(f2, s_vec));
                vst1q_f32(x + i + 12, vmulq_f32(f3, s_vec));
            } else {
                // Slow path crossing boundary (rare)
                for (int k = 0; k < 16; k++) x[i + k] = qx->q[i + k] * qx->s[(i + k) / group_size];
            }
        }
        for (; i < n; i++) {
            x[i] = qx->q[i] * qx->s[i / group_size];
        }
#elif defined(__AVX2__)
        int i = 0;
        for (; i <= n - 16; i += 16) {
            // Load 16 int8
            __m128i q_raw = _mm_loadu_si128(reinterpret_cast<const __m128i *>(qx->q + i));

            // Extend to 16-bit
            __m256i q_16 = _mm256_cvtepi8_epi16(q_raw);

            // Extend to 32-bit (low and high)
            __m256i q_32_lo = _mm256_cvtepi16_epi32(_mm256_castsi256_si128(q_16));
            __m256i q_32_hi = _mm256_cvtepi16_epi32(_mm256_extracti128_si256(q_16, 1));

            // Convert to float
            __m256 f_lo = _mm256_cvtepi32_ps(q_32_lo);
            __m256 f_hi = _mm256_cvtepi32_ps(q_32_hi);

            float s0 = qx->s[i / group_size];
            float s1 = qx->s[(i + 15) / group_size];

            if (s0 == s1) {
                __m256 s_vec = _mm256_set1_ps(s0);
                _mm256_storeu_ps(x + i, _mm256_mul_ps(f_lo, s_vec));
                _mm256_storeu_ps(x + i + 8, _mm256_mul_ps(f_hi, s_vec));
            } else {
                for (int k = 0; k < 16; k++) x[i + k] = qx->q[i + k] * qx->s[(i + k) / group_size];
            }
        }
        for (; i < n; i++) {
            x[i] = qx->q[i] * qx->s[i / group_size];
        }
#else
        for (int i = 0; i < n; i++) {
            x[i] = qx->q[i] * qx->s[i / group_size];
        }
#endif
    }

    void QuantizedTransformer::rmsnorm(float *__restrict__ o, const float *__restrict__ x,
                                       const float *__restrict__ weight, const int size) {
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
        __m128 sum_high = _mm256_extractf128_ps(sum_vec, 1);
        __m128 sum_low = _mm256_castps256_ps128(sum_vec);
        __m128 sum128 = _mm_add_ps(sum_low, sum_high);
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

        // Fast Inverse Square Root with Newton-Raphson refinement
#if defined(__ARM_NEON)
        const float32x4_t ss_v = vdupq_n_f32(ss);
        float32x4_t rsqrt_est = vrsqrteq_f32(ss_v);
        // Step: y = y * (3 - x * y * y) / 2
        // vrsqrtsq_f32(x, y) returns (3 - x * y) / 2 ... wait, no. 
        // ARM v8 manual: vrsqrts computes (3 - a * b) / 2.
        // So step is: est * vrsqrts(ss, est*est)
        const float32x4_t rsqrt_step = vrsqrtsq_f32(ss_v, vmulq_f32(rsqrt_est, rsqrt_est));
        rsqrt_est = vmulq_f32(rsqrt_est, rsqrt_step);
        ss = vgetq_lane_f32(rsqrt_est, 0);
#elif defined(__AVX2__)
        __m128 ss_v = _mm_set_ss(ss);
        __m128 rsqrt_est = _mm_rsqrt_ss(ss_v);
        // Newton-Raphson: y_n+1 = 0.5 * y_n * (3 - x * y_n^2)
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

    void QuantizedTransformer::quantize(const QuantizedTensor *qx, const float *x, const int n) const {
        const int gs = group_size;
        for (int i = 0; i < n; i += gs) {
            float max_val = 0.0f;
            const int current_gs = std::min(gs, n - i);

            for (int j = 0; j < current_gs; j++) {
                if (const float val = std::abs(x[i + j]); val > max_val) max_val = val;
            }

            const float scale = max_val / 127.0f;
            const float id = scale != 0.0f ? 1.0f / scale : 0.0f;

            qx->s[i / gs] = scale;

            for (int j = 0; j < current_gs; j++) {
                qx->q[i + j] = static_cast<int8_t>(std::round(x[i + j] * id));
            }
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
                        __builtin_prefetch(x->q + j + k + 128, 0, 0);
                        __builtin_prefetch(w->q + in0 + j + k + 128, 0, 0);
                        __builtin_prefetch(w->q + in1 + j + k + 128, 0, 0);
                        __builtin_prefetch(w->q + in2 + j + k + 128, 0, 0);
                        __builtin_prefetch(w->q + in3 + j + k + 128, 0, 0);

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
                    const int8_t qx = x->q[j + k];
                    ival0 += static_cast<int32_t>(qx) * static_cast<int32_t>(w->q[in0 + j + k]);
                    ival1 += static_cast<int32_t>(qx) * static_cast<int32_t>(w->q[in1 + j + k]);
                    ival2 += static_cast<int32_t>(qx) * static_cast<int32_t>(w->q[in2 + j + k]);
                    ival3 += static_cast<int32_t>(qx) * static_cast<int32_t>(w->q[in3 + j + k]);
                }

                const float sx = x->s[j / group_size];
                val0 += static_cast<float>(ival0) * w->s[(in0 + j) / group_size] * sx;
                val1 += static_cast<float>(ival1) * w->s[(in1 + j) / group_size] * sx;
                val2 += static_cast<float>(ival2) * w->s[(in2 + j) / group_size] * sx;
                val3 += static_cast<float>(ival3) * w->s[(in3 + j) / group_size] * sx;
                }
#elif defined(__AVX2__)
                for (int j = 0; j <= n - group_size; j += group_size) {
                    // Accumulators for 4 outputs
                    // We need to accumulate into 32-bit integers.
                    // Since group_size is typically 64 or 32, we process chunks.

                    __m256i sum0 = _mm256_setzero_si256();
                    __m256i sum1 = _mm256_setzero_si256();
                    __m256i sum2 = _mm256_setzero_si256();
                    __m256i sum3 = _mm256_setzero_si256();

                    int k = 0;
                    for (; k <= group_size - 32; k += 32) {
                        // Load 32 ints (int8) from x
                        __m256i qx_raw = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(x->q + j + k));

                        // Load 32 ints from w0..w3

                        _mm_prefetch(reinterpret_cast<const char *>(w->q + in0 + j + k + 128), _MM_HINT_T0);

                        _mm_prefetch(reinterpret_cast<const char *>(w->q + in1 + j + k + 128), _MM_HINT_T0);

                        _mm_prefetch(reinterpret_cast<const char *>(w->q + in2 + j + k + 128), _MM_HINT_T0);

                        _mm_prefetch(reinterpret_cast<const char *>(w->q + in3 + j + k + 128), _MM_HINT_T0);

                        _mm_prefetch(reinterpret_cast<const char *>(x->q + j + k + 128), _MM_HINT_T0);


                        __m256i qw0_raw = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(w->q + in0 + j + k));
                        __m256i qw1_raw = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(w->q + in1 + j + k));
                        __m256i qw2_raw = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(w->q + in2 + j + k));
                        __m256i qw3_raw = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(w->q + in3 + j + k));

                        // Split into low/high 128-bit lanes and sign extend to 16-bit
                        // _mm256_cvtepi8_epi16 takes lower 128 bits (16 bytes) and extends to 256 bits (16 words)
                        // So we need two ops for full 32 bytes.
                        // Wait, _mm256_cvtepi8_epi16 consumes __m128i.

                        __m128i qx_lo_128 = _mm256_castsi256_si128(qx_raw);
                        __m128i qx_hi_128 = _mm256_extracti128_si256(qx_raw, 1);

                        __m256i qx_lo = _mm256_cvtepi8_epi16(qx_lo_128);
                        __m256i qx_hi = _mm256_cvtepi8_epi16(qx_hi_128);

                        auto process_w = [&](__m256i w_raw, __m256i &acc) {
                            __m128i w_lo_128 = _mm256_castsi256_si128(w_raw);
                            __m128i w_hi_128 = _mm256_extracti128_si256(w_raw, 1);

                            __m256i w_lo = _mm256_cvtepi8_epi16(w_lo_128);
                            __m256i w_hi = _mm256_cvtepi8_epi16(w_hi_128);

                            // madd_epi16: (a0*b0 + a1*b1), ... returns 8x32-bit integers
                            __m256i prod_lo = _mm256_madd_epi16(qx_lo, w_lo);
                            __m256i prod_hi = _mm256_madd_epi16(qx_hi, w_hi);

                            // Accumulate
                            acc = _mm256_add_epi32(acc, prod_lo);
                            acc = _mm256_add_epi32(acc, prod_hi);
                        };

                        process_w(qw0_raw, sum0);
                        process_w(qw1_raw, sum1);
                        process_w(qw2_raw, sum2);
                        process_w(qw3_raw, sum3);
                    }

                    // Reduce sum0..sum3 (which are vectors of 8 partial sums)
                    auto hsum256_epi32 = [](__m256i v) -> int32_t {
                        // Horizontal sum of 8x32bit ints
                        __m128i lo = _mm256_castsi256_si128(v);
                        __m128i hi = _mm256_extracti128_si256(v, 1);
                        lo = _mm_add_epi32(lo, hi); // 4 ints
                        lo = _mm_hadd_epi32(lo, lo); // 2 ints (duplicated)
                        lo = _mm_hadd_epi32(lo, lo); // 1 int
                        return _mm_cvtsi128_si32(lo);
                    };

                    int32_t ival0 = hsum256_epi32(sum0);
                    int32_t ival1 = hsum256_epi32(sum1);
                    int32_t ival2 = hsum256_epi32(sum2);
                    int32_t ival3 = hsum256_epi32(sum3);

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
#elif defined(__AVX2__)
                    for (int j = 0; j <= n - group_size; j += group_size) {
                        __m256i sum = _mm256_setzero_si256();
                        int k = 0;
                        for (; k <= group_size - 32; k += 32) {
                            _mm_prefetch(reinterpret_cast<const char *>(w->q + in + j + k + 32), _MM_HINT_T0);
                            _mm_prefetch(reinterpret_cast<const char *>(x->q + j + k + 32), _MM_HINT_T0);
                            __m256i qx_raw = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(x->q + j + k));
                            __m256i qw_raw = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(w->q + in + j + k));

                            __m256i qx_lo = _mm256_cvtepi8_epi16(_mm256_castsi256_si128(qx_raw));
                            __m256i qx_hi = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(qx_raw, 1));

                            __m256i qw_lo = _mm256_cvtepi8_epi16(_mm256_castsi256_si128(qw_raw));
                            __m256i qw_hi = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(qw_raw, 1));

                            sum = _mm256_add_epi32(sum, _mm256_madd_epi16(qx_lo, qw_lo));
                            sum = _mm256_add_epi32(sum, _mm256_madd_epi16(qx_hi, qw_hi));
                        }

                        __m128i lo = _mm256_castsi256_si128(sum);
                        __m128i hi = _mm256_extracti128_si256(sum, 1);
                        lo = _mm_add_epi32(lo, hi);
                        lo = _mm_hadd_epi32(lo, lo);
                        lo = _mm_hadd_epi32(lo, lo);
                        int32_t ival = _mm_cvtsi128_si32(lo);

                        for (; k < group_size; k++) {
                            ival += static_cast<int32_t>(x->q[j + k]) * static_cast<int32_t>(w->q[in + j + k]);
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

    void QuantizedTransformer::run_layer(int l, int pos, float *x) {
        const Config *p = &config;
        const QuantizedTransformerWeights *w = &weights;
        QuantizedRunState *s = &state;
        const int dim = p->dim;
        const int kv_dim = (p->dim * p->n_kv_heads) / p->n_heads;
        const int kv_mul = p->n_heads / p->n_kv_heads;
        const int hidden_dim = p->hidden_dim;
        const int head_size = dim / p->n_heads;

        rmsnorm(s->xb.data(), x, w->rms_att_weight + l * dim, dim);

        // Construct temp QuantizedTensor for xq
        QuantizedTensor xq_tensor{s->xq_q.data(), s->xq_s.data()};
        quantize(&xq_tensor, s->xb.data(), dim);

        // Calculate pointers to the KV cache for this layer and position
        const int loff = l * p->seq_len * kv_dim;
        float *k_cache_ptr = s->key_cache.data() + loff + pos * kv_dim;
        float *v_cache_ptr = s->value_cache.data() + loff + pos * kv_dim;

        // Compute Q, K, V
        matmul(s->q.data(), &xq_tensor, &w->wq[l], dim, dim);
        matmul(k_cache_ptr, &xq_tensor, &w->wk[l], dim, kv_dim);
        matmul(v_cache_ptr, &xq_tensor, &w->wv[l], dim, kv_dim);

        // RoPE
        const int rope_offset = pos * (head_size / 2);
#pragma omp parallel for
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
                    float32x4_t k_vec = vld1q_f32(k_cache_ptr + i * head_size + j);
                    float32x4_t k_term1 = vmulq_f32(k_vec, c_vec);
                    float32x4_t k_swap = vrev64q_f32(k_vec);
                    float32x4_t k_term2 = vmulq_f32(k_swap, s_signed);
                    float32x4_t k_res = vaddq_f32(k_term1, k_term2);
                    vst1q_f32(k_cache_ptr + i * head_size + j, k_res);
                }
            }
#elif defined(__AVX2__)
            for (; j <= head_size - 8; j += 8) {
                __m256 q_vec = _mm256_loadu_ps(s->q.data() + i * head_size + j);

                __m128 c_small = _mm_loadu_ps(&rope_cos[rope_offset + j / 2]);
                __m128 s_small = _mm_loadu_ps(&rope_sin[rope_offset + j / 2]);

                __m128 c_lo = _mm_unpacklo_ps(c_small, c_small);
                __m128 c_hi = _mm_unpackhi_ps(c_small, c_small);
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
                    __m256 k_vec = _mm256_loadu_ps(k_cache_ptr + i * head_size + j);
                    __m256 k_swap = _mm256_permute_ps(k_vec, _MM_SHUFFLE(2, 3, 0, 1));
                    __m256 k_t1 = _mm256_mul_ps(k_vec, c_vec);
                    __m256 k_t2 = _mm256_mul_ps(k_swap, s_vec);
                    __m256 k_res = _mm256_addsub_ps(k_t1, k_t2);
                    _mm256_storeu_ps(k_cache_ptr + i * head_size + j, k_res);
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
                    const float k0 = k_cache_ptr[i * head_size + j];
                    const float k1 = k_cache_ptr[i * head_size + j + 1];
                    k_cache_ptr[i * head_size + j] = k0 * fcr - k1 * fci;
                    k_cache_ptr[i * head_size + j + 1] = k0 * fci + k1 * fcr;
                }
            }
        }

        int h;
#pragma omp parallel for private(h)
        for (h = 0; h < p->n_heads; h++) {
            const float *q = s->q.data() + h * head_size;
            float *att = s->att.data() + h * p->seq_len;
            int t = 0;

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

            for (; t <= pos; t++) {
                const float *k = s->key_cache.data() + loff + t * kv_dim + (h / kv_mul) * head_size;

                float score = 0.0f;
                int i = 0;
#if defined(__ARM_NEON)
                float32x4_t sum_vec = vdupq_n_f32(0.0f);
                for (; i <= head_size - 4; i += 4) {
                    float32x4_t q_vec = vld1q_f32(q + i);
                    float32x4_t k_vec = vld1q_f32(k + i);
                    sum_vec = vmlaq_f32(sum_vec, q_vec, k_vec);
                }
                score = vaddvq_f32(sum_vec);
#elif defined(__AVX2__)
                __m256 sum_vec = _mm256_setzero_ps();
                for (; i <= head_size - 8; i += 8) {
                    __m256 q_vec = _mm256_loadu_ps(q + i);
                    __m256 k_vec = _mm256_loadu_ps(k + i);
                    sum_vec = _mm256_fmadd_ps(q_vec, k_vec, sum_vec);
                }
                __m128 sum_high = _mm256_extractf128_ps(sum_vec, 1);
                __m128 sum_low = _mm256_castps256_ps128(sum_vec);
                __m128 sum128 = _mm_add_ps(sum_low, sum_high);
                sum128 = _mm_hadd_ps(sum128, sum128);
                sum128 = _mm_hadd_ps(sum128, sum128);
                score = _mm_cvtss_f32(sum128);
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
                    float32x4_t v_vec = vld1q_f32(v + i);
                    xb_vec = vmlaq_f32(xb_vec, a_vec, v_vec);
                    vst1q_f32(xb + i, xb_vec);
                }
#elif defined(__AVX2__)
                __m256 a_vec = _mm256_set1_ps(a);
                for (; i <= head_size - 8; i += 8) {
                    __m256 xb_vec = _mm256_loadu_ps(xb + i);
                    __m256 v_vec = _mm256_loadu_ps(v + i);
                    xb_vec = _mm256_fmadd_ps(a_vec, v_vec, xb_vec);
                    _mm256_storeu_ps(xb + i, xb_vec);
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

#pragma omp parallel for simd
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

    float *QuantizedTransformer::forward(const int token, const int pos, const int flags) {
        const Config *p = &config;
        const QuantizedTransformerWeights *w = &weights;
        QuantizedRunState *s = &state;
        float *x = s->x.data();
        const int dim = p->dim;

        std::copy_n(w->token_embedding_table.data() + token * dim, dim, x);

        // --- Load Balancing Trigger (Master Only) ---
        if (dist_config.mode == DistributedMode::Master && pos % 50 == 0 && pos > 0) {
            std::vector<NodeStats> node_stats = collect_stats();
            size_t recv_count = node_stats.size();
            
            if (recv_count > 1) { 
                std::vector<float> scores(recv_count);
                for (size_t i = 0; i < recv_count; i++) {
                    float s = node_stats[i].cpu_usage;
                    if (node_stats[i].temperature > 80.0f) s += 20.0f; 
                    scores[i] = s;
                }

                bool changed = false;
                std::vector<LayerConfig> new_configs(recv_count);
                for(size_t i=0; i<recv_count; i++) new_configs[i] = {node_stats[i].split_layer, node_stats[i].end_layer};

                for (size_t i = 0; i < recv_count - 1; i++) {
                    float diff = scores[i] - scores[i+1];
                    const float threshold = 25.0f; 

                    if (std::abs(diff) > threshold) {
                        if (diff > 0) {
                            int layers_i = new_configs[i].end_layer - new_configs[i].split_layer;
                            if (layers_i > 1) { 
                                new_configs[i].end_layer--;
                                new_configs[i+1].split_layer--;
                                changed = true;
                                std::cout << "[LB] Imbalance detected (Node " << i << " > Node " << i+1 << "). Shifting 1 layer >>" << std::endl;
                                break; 
                            }
                        } else {
                            int layers_next = new_configs[i+1].end_layer - new_configs[i+1].split_layer;
                            if (layers_next > 1) {
                                new_configs[i].end_layer++;
                                new_configs[i+1].split_layer++;
                                changed = true;
                                std::cout << "[LB] Imbalance detected (Node " << i+1 << " > Node " << i << "). Shifting 1 layer <<" << std::endl;
                                break;
                            }
                        }
                    }
                }

                if (changed) {
                    distribute_config(new_configs);
                    needs_rewind = true;
                    return s->logits.data();
                }
            }
        }
        // ---------------------------------------------

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

            const PacketHeader header{0x4C454150, PacketType::Inference, (uint32_t)pos, 1, (uint32_t)flags, (uint32_t)(dim * sizeof(float))};

            // Optimization: Zero-Copy Send
            dist_config.transport->send_multipart_next(&header, sizeof(PacketHeader), x, dim * sizeof(float));

            if (flags == FLAG_NEED_REPLY) {
                // Ring Synchronization (Stop-and-Wait):
                const size_t packet_size = sizeof(PacketHeader) + dim * sizeof(float);
                if (transfer_buffer.size() < packet_size) transfer_buffer.resize(packet_size);

                dist_config.transport->recv_prev(transfer_buffer.data(), packet_size);

                // Extract the data (skip header)
                std::memcpy(x, transfer_buffer.data() + sizeof(PacketHeader), dim * sizeof(float));
            } else {
                return nullptr;
            }
        }

        rmsnorm(x, x, w->rms_final_weight, dim);
        const QuantizedTensor xq_tensor{s->xq_q.data(), s->xq_s.data()};
        quantize(&xq_tensor, x, dim);
        matmul(s->logits.data(), &xq_tensor, w->wcls, dim, p->vocab_size);
        return s->logits.data();
    }

    void QuantizedTransformer::forward_batch(const std::vector<int> &tokens, int start_pos) {
        const Config *p = &config;
        const QuantizedTransformerWeights *w = &weights;
        const int dim = p->dim;
        size_t n_tokens = tokens.size();

        std::vector<float> batch_x(n_tokens * dim);

        int start_layer = 0;
        int end_layer = p->n_layers;
        if (dist_config.mode == DistributedMode::Master) {
            end_layer = dist_config.split_layer;
        }

        for (size_t i = 0; i < n_tokens; i++) {
            float *x_ptr = batch_x.data() + i * dim;
            std::copy_n(w->token_embedding_table.data() + tokens[i] * dim, dim, x_ptr);
            for (int l = start_layer; l < end_layer; l++) {
                run_layer(l, start_pos + i, x_ptr);
            }
        }

        if (dist_config.mode == DistributedMode::Master) {
            if (!dist_config.transport) throw std::runtime_error("Transport not set for master");
            PacketHeader header{0x4C454150, PacketType::Inference, (uint32_t)start_pos, (uint32_t)n_tokens, FLAG_NO_REPLY};
            header.payload_size = n_tokens * dim * sizeof(float);
            dist_config.transport->send_multipart_next(&header, sizeof(PacketHeader), batch_x.data(), header.payload_size);
        }
    }

    void QuantizedTransformer::worker_loop() {
        if (!dist_config.transport) throw std::runtime_error("Transport not set for worker");

        float *x = state.x.data();
        const int dim = config.dim;
        PacketHeader header{};

        const size_t max_payload = dim * sizeof(float) > 1024 ? dim * sizeof(float) : 1024;
        if (transfer_buffer.size() < sizeof(PacketHeader) + max_payload) 
            transfer_buffer.resize(sizeof(PacketHeader) + max_payload);

        std::cout << "Worker started. Processing layers " << dist_config.split_layer << " to " << dist_config.end_layer - 1 <<
                std::endl;

        while (true) {
            try {
                // 1. Receive Header Only
                dist_config.transport->recv_prev(transfer_buffer.data(), sizeof(PacketHeader));
                std::memcpy(&header, transfer_buffer.data(), sizeof(PacketHeader));

                if (header.magic != 0x4C454150) {
                     std::cerr << "Error: Invalid Magic " << std::hex << header.magic << std::dec << std::endl;
                     break;
                }

                // 2. Receive Payload (if any)
                if (header.payload_size > 0) {
                     if (transfer_buffer.size() < sizeof(PacketHeader) + header.payload_size)
                         transfer_buffer.resize(sizeof(PacketHeader) + header.payload_size);
                     
                     dist_config.transport->recv_prev(transfer_buffer.data() + sizeof(PacketHeader), header.payload_size);
                }

                if (header.type == PacketType::Inference) {
                    size_t n_tokens = header.n_tokens > 0 ? header.n_tokens : 1;
                    size_t data_size = n_tokens * dim * sizeof(float);
                    float *data_ptr = reinterpret_cast<float*>(transfer_buffer.data() + sizeof(PacketHeader));

                    for (size_t i = 0; i < n_tokens; i++) {
                        float *token_data = data_ptr + i * dim;
                        for (int l = dist_config.split_layer; l < dist_config.end_layer; l++) {
                            run_layer(l, header.pos + i, token_data);
                        }
                    }

                    if (!dist_config.is_tail || header.flags == FLAG_NEED_REPLY) {
                        dist_config.transport->send_multipart_next(&header, sizeof(PacketHeader), data_ptr, data_size);
                    }
                } 
                else if (header.type == PacketType::StatsUpdate) {
                    uint32_t count;
                    std::memcpy(&count, transfer_buffer.data() + sizeof(PacketHeader), sizeof(count));
                    NodeStats my_stats = monitor.get_stats();
                    my_stats.split_layer = dist_config.split_layer;
                    my_stats.end_layer = dist_config.end_layer;

                    size_t offset = sizeof(uint32_t) + count * sizeof(NodeStats);
                    if (offset + sizeof(NodeStats) <= header.payload_size) {
                        std::memcpy(transfer_buffer.data() + sizeof(PacketHeader) + offset, &my_stats, sizeof(NodeStats));
                        count++;
                        std::memcpy(transfer_buffer.data() + sizeof(PacketHeader), &count, sizeof(count));
                    }
                    dist_config.transport->send_next(transfer_buffer.data(), sizeof(PacketHeader) + header.payload_size);
                }
                else if (header.type == PacketType::ConfigUpdate) {
                    uint32_t current_idx;
                    std::memcpy(&current_idx, transfer_buffer.data() + sizeof(PacketHeader), sizeof(current_idx));
                    LayerConfig *configs = reinterpret_cast<LayerConfig*>(transfer_buffer.data() + sizeof(PacketHeader) + sizeof(uint32_t));
                    LayerConfig my_cfg = configs[current_idx];
                    update_config(my_cfg.split_layer, my_cfg.end_layer);
                    current_idx++;
                    std::memcpy(transfer_buffer.data() + sizeof(PacketHeader), &current_idx, sizeof(current_idx));
                    dist_config.transport->send_next(transfer_buffer.data(), sizeof(PacketHeader) + header.payload_size);
                }
            } catch (const std::exception &e) {
                std::cerr << "Worker loop error: " << e.what() << std::endl;
                break;
            }
        }
    }

    void QuantizedTransformer::distribute_config(const std::vector<LayerConfig> &configs) {
        if (dist_config.mode != DistributedMode::Master || configs.empty()) return;

        PacketHeader req{};
        req.magic = 0x4C454150;
        req.type = PacketType::ConfigUpdate;
        req.payload_size = sizeof(uint32_t) + configs.size() * sizeof(LayerConfig);
        req.pos = 0;
        req.flags = FLAG_NEED_REPLY;

        std::vector<char> buf(req.payload_size);
        update_config(configs[0].split_layer, configs[0].end_layer);
        uint32_t start_idx = 1;
        std::memcpy(buf.data(), &start_idx, sizeof(start_idx));
        std::memcpy(buf.data() + sizeof(start_idx), configs.data(), configs.size() * sizeof(LayerConfig));

        dist_config.transport->send_multipart_next(&req, sizeof(PacketHeader), buf.data(), buf.size());
        if (transfer_buffer.size() < sizeof(PacketHeader) + req.payload_size)
            transfer_buffer.resize(sizeof(PacketHeader) + req.payload_size);
        dist_config.transport->recv_prev(transfer_buffer.data(), sizeof(PacketHeader) + req.payload_size);
        std::cout << "[Master] Configuration distributed to cluster." << std::endl;
    }

    std::vector<NodeStats> QuantizedTransformer::collect_stats() {
        if (dist_config.mode != DistributedMode::Master) return {};

        PacketHeader req{};
        req.magic = 0x4C454150;
        req.type = PacketType::StatsUpdate;
        req.payload_size = sizeof(uint32_t) + 32 * sizeof(NodeStats); 
        req.pos = 0;
        req.flags = FLAG_NEED_REPLY;

        std::vector<char> stats_buf(req.payload_size);
        uint32_t count = 1;
        NodeStats my_stats = monitor.get_stats();
        my_stats.split_layer = dist_config.split_layer;
        my_stats.end_layer = dist_config.end_layer;
        
        std::memcpy(stats_buf.data(), &count, sizeof(count));
        std::memcpy(stats_buf.data() + sizeof(count), &my_stats, sizeof(NodeStats));

        dist_config.transport->send_multipart_next(&req, sizeof(PacketHeader), stats_buf.data(), stats_buf.size());

        if (transfer_buffer.size() < sizeof(PacketHeader) + req.payload_size)
            transfer_buffer.resize(sizeof(PacketHeader) + req.payload_size);
            
        dist_config.transport->recv_prev(transfer_buffer.data(), sizeof(PacketHeader) + req.payload_size);
        
        uint32_t recv_count;
        std::memcpy(&recv_count, transfer_buffer.data() + sizeof(PacketHeader), sizeof(recv_count));
        
        std::vector<NodeStats> node_stats(recv_count);
        if (recv_count > 0) {
            std::memcpy(node_stats.data(), transfer_buffer.data() + sizeof(PacketHeader) + sizeof(uint32_t), recv_count * sizeof(NodeStats));
        }
        return node_stats;
    }

    void QuantizedTransformer::clear_cache() {
        std::fill(state.key_cache.begin(), state.key_cache.end(), 0.0f);
        std::fill(state.value_cache.begin(), state.value_cache.end(), 0.0f);
    }
} // namespace Inference