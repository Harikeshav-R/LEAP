#ifndef LEAP_TOKENIZER_H
#define LEAP_TOKENIZER_H

#include <vector>
#include <string>
#include <unordered_set>
#include <memory>
#include <pytorch/tokenizers/tiktoken.h>

namespace Tokenizer {
    class Tokenizer {
    public:
        explicit Tokenizer(const std::string &model_path);

        [[nodiscard]] std::vector<int> encode(const std::string &s, bool bos, bool eos) const;

        [[nodiscard]] std::string decode(const std::vector<int> &tokens) const;

        void export_tokenized_binary_file(const std::string& output_path_arg = "") const;

        [[nodiscard]] int32_t bos_id() const { return static_cast<int32_t>(bos_id_); }
        [[nodiscard]] int32_t eos_id() const { return static_cast<int32_t>(eos_id_); }
        [[nodiscard]] int32_t pad_id() const { return pad_id_; }
        [[nodiscard]] const std::unordered_set<uint64_t> &stop_tokens() const { return stop_tokens_; }

    private:
        std::unique_ptr<tokenizers::Tiktoken> model_;
        std::string model_path_;
        uint64_t n_words_;
        uint64_t bos_id_;
        uint64_t eos_id_;
        int32_t pad_id_;
        std::unordered_set<uint64_t> stop_tokens_;
    };
}

#endif //LEAP_TOKENIZER_H