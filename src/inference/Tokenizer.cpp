#include "Tokenizer.h"
#include <stdexcept>
#include <iostream>
#include <fstream>
#include <algorithm>
#include <vector>

namespace Inference {
    Tokenizer::Tokenizer(const std::string &tokenizer_path, const int vocab_size) : vocab_size(vocab_size) {
        // Initialize byte pieces
        byte_pieces.resize(256);
        for (int i = 0; i < 256; i++) {
            byte_pieces[i].push_back(static_cast<char>(i));
        }

        std::ifstream file(tokenizer_path, std::ios::binary);
        if (!file.is_open()) {
            const std::string assets_path = "assets/" + tokenizer_path;
            file.open(assets_path, std::ios::binary);
            if (!file.is_open()) {
                throw std::runtime_error("couldn't load " + tokenizer_path);
            }
        }

        if (!file.read(reinterpret_cast<char *>(&max_token_length), sizeof(int))) {
            throw std::runtime_error("failed read max_token_length");
        }

        vocab.resize(vocab_size);
        vocab_scores.resize(vocab_size);

        for (int i = 0; i < vocab_size; i++) {
            float score;
            if (!file.read(reinterpret_cast<char *>(&score), sizeof(float))) {
                throw std::runtime_error("failed read vocab_scores");
            }
            vocab_scores[i] = score;

            int len;
            if (!file.read(reinterpret_cast<char *>(&len), sizeof(int))) {
                throw std::runtime_error("failed read len");
            }

            std::string str(len, '\0');
            if (!file.read(&str[0], len)) {
                throw std::runtime_error("failed read vocab string");
            }
            vocab[i] = std::move(str);
        }
    }

    const std::string &Tokenizer::decode(int prev_token, const int token) const {
        const std::string &piece = vocab[token];
        // careful, some tokens designate raw bytes, and look like e.g. '<0x01>'
        if (piece.size() == 6 && piece.substr(0, 3) == "<0x" && piece.back() == '>') {
            try {
                const int byte_val = std::stoi(piece.substr(3, 2), nullptr, 16);
                return byte_pieces[byte_val];
            } catch (...) {
                // Fallback if parsing fails
            }
        }
        return piece;
    }

    int Tokenizer::str_lookup(const std::string &str, const std::vector<TokenIndex> &sorted_vocab) {
        // Use binary search (std::lower_bound)
        TokenIndex target{};
        target.str = &str;
        const auto it = std::ranges::lower_bound(sorted_vocab, target,
                                                 [](const TokenIndex &a, const TokenIndex &b) {
                                                     return *a.str < *b.str;
                                                 });

        if (it != sorted_vocab.end() && *it->str == str) {
            return it->id;
        }
        return -1;
    }

    void Tokenizer::encode(const std::string &text, const bool bos, const bool eos, std::vector<int> &tokens,
                           int &n_tokens) {
        if (text.empty()) {
            n_tokens = 0;
            return;
        }

        if (sorted_vocab.empty()) {
            sorted_vocab.resize(vocab_size);
            for (int i = 0; i < vocab_size; i++) {
                sorted_vocab[i].str = &vocab[i];
                sorted_vocab[i].id = i;
            }
            std::ranges::sort(sorted_vocab, [](const TokenIndex &a, const TokenIndex &b) {
                return *a.str < *b.str;
            });
        }

        std::string str_buffer;
        str_buffer.reserve(max_token_length * 2 + 10);

        n_tokens = 0;
        tokens.clear();
        tokens.resize(text.size() + 3);

        if (bos)
            tokens[n_tokens++] = 128000;

        str_buffer.clear();
        for (size_t k = 0; k < text.length(); k++) {
            const char c = text[k];
            if ((c & 0xC0) != 0x80) {
                str_buffer.clear();
            }
            str_buffer.push_back(c);

            if (k + 1 < text.length()) {
                if (const char next_c = text[k + 1]; (next_c & 0xC0) == 0x80 && str_buffer.length() < 4) {
                    continue;
                }
            }

            if (const int id = str_lookup(str_buffer, sorted_vocab); id != -1) {
                tokens[n_tokens++] = id;
            } else {
                for (const char b: str_buffer) {
                    tokens[n_tokens++] = static_cast<unsigned char>(b) + 3;
                }
            }
            str_buffer.clear();
        }

        // BPE merge
        while (true) {
            float best_score = -1e10;
            int best_id = -1;
            int best_idx = -1;
            int best_len = 2;

            for (int i = 0; i < (n_tokens - 1); i++) {
                std::string merged = vocab[tokens[i]] + vocab[tokens[i + 1]];

                if (const int id = str_lookup(merged, sorted_vocab); id != -1 && vocab_scores[id] > best_score) {
                    best_score = vocab_scores[id];
                    best_id = id;
                    best_idx = i;
                }
            }

            if (best_idx == -1) {
                for (int i = 0; i < (n_tokens - 2); i++) {
                    std::string merged = vocab[tokens[i]] + vocab[tokens[i + 1]] + vocab[tokens[i + 2]];

                    if (const int id = str_lookup(merged, sorted_vocab); id != -1 && vocab_scores[id] > best_score) {
                        best_score = vocab_scores[id];
                        best_id = id;
                        best_idx = i;
                        best_len = 3;
                    }
                }
            }

            if (best_idx == -1) {
                break;
            }

            tokens[best_idx] = best_id;
            for (int i = best_idx + 1; i < (n_tokens - best_len + 1); i++) {
                tokens[i] = tokens[i + best_len - 1];
            }
            n_tokens -= (best_len - 1);
        }

        if (eos)
            tokens[n_tokens++] = 128001;

        tokens.resize(n_tokens);
    }
} // namespace Inference