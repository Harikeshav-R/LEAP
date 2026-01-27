#ifndef LEAP_TOKENIZER_H
#define LEAP_TOKENIZER_H

#include <string>
#include <vector>
#include <unordered_map>

namespace Inference {
    class Tokenizer {
    public:
        Tokenizer(const std::string &tokenizer_path, int vocab_size);

        ~Tokenizer() = default;

        const std::string &decode(int prev_token, int token) const;

        void encode(const std::string &text, bool bos, bool eos, std::vector<int> &tokens, int &n_tokens);

    private:
        std::vector<std::string> vocab;
        std::vector<float> vocab_scores;
        std::unordered_map<std::string_view, int> vocab_lookup;
        int vocab_size;
        unsigned int max_token_length{};
        std::vector<std::string> byte_pieces; // stores all single-byte strings

        int str_lookup(std::string_view str);
    };
} // namespace Inference

#endif // LEAP_TOKENIZER_H