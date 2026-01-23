#ifndef LEAP_TOKENIZER_H
#define LEAP_TOKENIZER_H

#include <string>
#include <vector>

namespace Inference {
    struct TokenIndex {
        const std::string *str;
        int id;
    };

    class Tokenizer {
    public:
        Tokenizer(const std::string &tokenizer_path, int vocab_size);

        ~Tokenizer() = default;

        const std::string &decode(int prev_token, int token) const;

        void encode(const std::string &text, bool bos, bool eos, std::vector<int> &tokens, int &n_tokens);

    private:
        std::vector<std::string> vocab;
        std::vector<float> vocab_scores;
        std::vector<TokenIndex> sorted_vocab; // array of size vocab_size
        int vocab_size;
        unsigned int max_token_length{};
        std::vector<std::string> byte_pieces; // stores all single-byte strings

        static int str_lookup(const std::string &str, const std::vector<TokenIndex> &sorted_vocab);
    };
} // namespace Inference

#endif // LEAP_TOKENIZER_H