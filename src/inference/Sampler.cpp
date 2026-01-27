#include "Sampler.h"
#include "Utils.h"
#include <cmath>
#include <algorithm>
#include <vector>
#include <iterator>

namespace Inference {
    Sampler::Sampler(const int vocab_size, const float temperature, const float topp, const unsigned long long rng_seed)
        : vocab_size(vocab_size), temperature(temperature), topp(topp), rng_state(rng_seed) {
        probindex.resize(vocab_size);
    }

    int Sampler::sample_argmax(const float *probabilities, const int n) {
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