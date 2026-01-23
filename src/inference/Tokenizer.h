#ifndef LEAP_TOKENIZER_H
#define LEAP_TOKENIZER_H

#include <string>
#include <vector>
#include <algorithm>

namespace Inference {
    struct TokenIndex {
        const char *str;
        int id;
    };

    class Tokenizer {
    public:
        Tokenizer(const std::string &tokenizer_path, int vocab_size);

        ~Tokenizer();

        const char *decode(int prev_token, int token) const;

        void encode(const std::string &text, bool bos, bool eos, std::vector<int> &tokens, int &n_tokens);

    private:
        char **vocab;
        float *vocab_scores;
        TokenIndex *sorted_vocab; // array of size vocab_size
        int vocab_size;
        unsigned int max_token_length{};
        unsigned char byte_pieces[512]{}; // stores all single-byte strings

        static int compare_tokens(const void *a, const void *b);

        static int str_lookup(const char *str, const TokenIndex *sorted_vocab, int vocab_size);
    };
} // namespace Inference

#endif // LEAP_TOKENIZER_H