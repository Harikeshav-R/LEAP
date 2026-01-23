#ifndef LEAP_SAMPLER_H
#define LEAP_SAMPLER_H

#include <vector>

namespace Inference {
    struct ProbIndex {
        float prob;
        int index;
    };

    class Sampler {
    public:
        Sampler(int vocab_size, float temperature, float topp, unsigned long long rng_seed);

        ~Sampler() = default;

        int sample(float *logits);

    private:
        int vocab_size;
        std::vector<ProbIndex> probindex; // buffer used in top-p sampling
        float temperature;
        float topp;
        unsigned long long rng_state;

        static int sample_argmax(const float *probabilities, int n);

        static int sample_mult(const float *probabilities, int n, float coin);

        static int sample_topp(const float *probabilities, int n, float topp, std::vector<ProbIndex> &probindex,
                               float coin);
    };
} // namespace Inference

#endif // LEAP_SAMPLER_H