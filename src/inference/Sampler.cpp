#include "Sampler.h"
#include "Utils.h"
#include <cmath>
#include <algorithm>
#include <vector>
#include <iterator>

#if defined(__x86_64__) || defined(_M_X64)
#include <immintrin.h>
#endif

namespace Inference {
    Sampler::Sampler(const int vocab_size, const float temperature, const float topp, const unsigned long long rng_seed)
        : vocab_size(vocab_size), temperature(temperature), topp(topp), rng_state(rng_seed) {
        probindex.resize(vocab_size);
    }

    int Sampler::sample_argmax(const float *probabilities, const int n) {
#if defined(__AVX2__)
        // AVX2 vectorized argmax
        if (n >= 8) {
            float max_val = -1e30f;
            int max_idx = 0;

            // Initial scalar pass for max value (or use reduction) is simpler for argmax
            // But we need the index.
            // Vectorized approach: keep max_val_vec and max_idx_vec.

            __m256 max_v = _mm256_set1_ps(-1e30f);
            __m256i max_idx_v = _mm256_setzero_si256();

            // Current indices
            __m256i idx_v = _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7);
            const __m256i inc_v = _mm256_set1_epi32(8);

            int i = 0;
            for (; i <= n - 8; i += 8) {
                __m256 val = _mm256_loadu_ps(probabilities + i);

                // Compare val > max_v
                __m256 mask = _mm256_cmp_ps(val, max_v, _CMP_GT_OQ);

                // Update max_v using mask
                max_v = _mm256_blendv_ps(max_v, val, mask);

                // Update max_idx_v using mask (cast mask to integer)
                // blendv requires matching types. _mm256_castps_si256(mask) is not enough for blendv_epi8.
                // Use _mm256_blendv_epi8 (AVX2). It operates on bytes.
                // We need 32-bit blend.
                // Unfortunately _mm256_blendv_epi32 doesn't exist? _mm256_blendv_epi8 exists.
                // We can use it.

                __m256i mask_i = _mm256_castps_si256(mask);
                max_idx_v = _mm256_blendv_epi8(max_idx_v, idx_v, mask_i);

                idx_v = _mm256_add_epi32(idx_v, inc_v);
            }

            // Reduce max_v and max_idx_v
            float vals[8];
            int idxs[8];
            _mm256_storeu_ps(vals, max_v);
            _mm256_storeu_si256(reinterpret_cast<__m256i *>(idxs), max_idx_v);

            max_val = vals[0];
            max_idx = idxs[0];

            for (int k = 1; k < 8; ++k) {
                if (vals[k] > max_val) {
                    max_val = vals[k];
                    max_idx = idxs[k];
                } else if (vals[k] == max_val && idxs[k] < max_idx) {
                    // Stability check, prefer lower index on tie (std::max_element behavior)
                    max_idx = idxs[k];
                }
            }

            // Tail cleanup
            for (; i < n; ++i) {
                if (probabilities[i] > max_val) {
                    max_val = probabilities[i];
                    max_idx = i;
                }
            }
            return max_idx;
        }
#endif
        const auto it = std::max_element(probabilities, probabilities + n);
        return static_cast<int>(std::distance(probabilities, it));
    }

    int Sampler::sample_mult(const float *probabilities, const int n, const float coin) {
        float cdf = 0.0f;
        for (int i = 0; i < n; i++) {
            cdf += probabilities[i];
            if (coin < cdf) {
                return i;
            }
        }
        return n - 1;
    }

    int Sampler::sample_topp(const float *probabilities, const int n, const float topp,
                             std::vector<ProbIndex> &probindex,
                             const float coin) {
        int n0 = 0;
        const float cutoff = (1.0f - topp) / (static_cast<float>(n) - 1.0f);

        // This loop selects candidates. Can use std::copy_if but manual is fine for speed/simplicity here
        for (int i = 0; i < n; i++) {
            if (probabilities[i] >= cutoff) {
                probindex[n0].index = i;
                probindex[n0].prob = probabilities[i];
                n0++;
            }
        }

        std::sort(probindex.begin(), probindex.begin() + n0, [](const ProbIndex &a, const ProbIndex &b) {
            return a.prob > b.prob;
        });

        float cumulative_prob = 0.0f;
        int last_idx = n0 - 1;
        for (int i = 0; i < n0; i++) {
            cumulative_prob += probindex[i].prob;
            if (cumulative_prob > topp) {
                last_idx = i;
                break;
            }
        }

        const float r = coin * cumulative_prob;
        float cdf = 0.0f;
        for (int i = 0; i <= last_idx; i++) {
            cdf += probindex[i].prob;
            if (r < cdf) {
                return probindex[i].index;
            }
        }
        return probindex[last_idx].index;
    }

    static void softmax(float *x, const int size) {
        // 1. Find max for numerical stability
        float max_val = x[0];
#pragma omp parallel for reduction(max:max_val)
        for (int i = 1; i < size; ++i) {
            if (x[i] > max_val) {
                max_val = x[i];
            }
        }

        // 2. Exp and Sum
        float sum = 0.0f;
#pragma omp parallel for reduction(+:sum)
        for (int i = 0; i < size; i++) {
            float val = std::exp(x[i] - max_val);
            x[i] = val;
            sum += val;
        }

        // 3. Normalize
        const float inv_sum = 1.0f / sum;
#pragma omp parallel for
        for (int i = 0; i < size; i++) {
            x[i] *= inv_sum;
        }
    }

    int Sampler::sample(float *logits) {
        int next;
        if (temperature == 0.0f) {
            next = sample_argmax(logits, vocab_size);
        } else {
            for (int q = 0; q < vocab_size; q++) {
                logits[q] /= temperature;
            }
            softmax(logits, vocab_size);
            const float coin = Utils::random_f32(rng_state);
            if (topp <= 0 || topp >= 1) {
                next = sample_mult(logits, vocab_size, coin);
            } else {
                next = sample_topp(logits, vocab_size, topp, probindex, coin);
            }
        }
        return next;
    }
} // namespace Inference