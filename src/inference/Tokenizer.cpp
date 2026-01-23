#include "Tokenizer.h"
#include <cstdio>
#include <stdexcept>
#include <iostream>

namespace Inference {
    Tokenizer::Tokenizer(const std::string &tokenizer_path, const int vocab_size) : vocab_size(vocab_size) {
        // malloc space to hold the scores and the strings
        vocab = static_cast<char **>(malloc(vocab_size * sizeof(char *)));
        vocab_scores = static_cast<float *>(malloc(vocab_size * sizeof(float)));
        sorted_vocab = nullptr; // initialized lazily

        for (int i = 0; i < 256; i++) {
            byte_pieces[i * 2] = static_cast<unsigned char>(i);
            byte_pieces[i * 2 + 1] = '\0';
        }

        // read in the file
        FILE *file = fopen(tokenizer_path.c_str(), "rb");
        if (!file) {
            // try looking in assets/
            const std::string assets_path = "assets/" + tokenizer_path;
            file = fopen(assets_path.c_str(), "rb");
            if (!file) {
                throw std::runtime_error("couldn't load " + tokenizer_path);
            }
        }

        if (fread(&max_token_length, sizeof(int), 1, file) != 1) {
            fclose(file);
            throw std::runtime_error("failed read max_token_length");
        }

        int len;
        for (int i = 0; i < vocab_size; i++) {
            if (fread(vocab_scores + i, sizeof(float), 1, file) != 1) {
                fclose(file);
                throw std::runtime_error("failed read vocab_scores");
            }
            if (fread(&len, sizeof(int), 1, file) != 1) {
                fclose(file);
                throw std::runtime_error("failed read len");
            }
            vocab[i] = static_cast<char *>(malloc(len + 1));
            if (fread(vocab[i], len, 1, file) != 1) {
                fclose(file);
                throw std::runtime_error("failed read vocab string");
            }
            vocab[i][len] = '\0'; // add the string terminating token
        }
        fclose(file);
    }

    Tokenizer::~Tokenizer() {
        for (int i = 0; i < vocab_size; i++) {
            free(vocab[i]);
        }
        free(vocab);
        free(vocab_scores);
        if (sorted_vocab) {
            free(sorted_vocab);
        }
    }

    int Tokenizer::compare_tokens(const void *a, const void *b) {
        return strcmp((static_cast<const TokenIndex *>(a))->str, (static_cast<const TokenIndex *>(b))->str);
    }

    const char *Tokenizer::decode(int prev_token, const int token) const {
        const char *piece = vocab[token];
        // careful, some tokens designate raw bytes, and look like e.g. '<0x01>'
        // parse this and convert and return the actual byte
        unsigned char byte_val;
        if (sscanf(piece, "<0x%02hhX>", &byte_val) == 1) {
            // Use reinterpret_cast to convert unsigned char* to char*
            piece = reinterpret_cast<const char *>(byte_pieces) + byte_val * 2;
        }
        return piece;
    }

    int Tokenizer::str_lookup(const char *str, const TokenIndex *sorted_vocab, const int vocab_size) {
        const TokenIndex tok = {.str = str}; // acts as the key to search for
        const auto *res = static_cast<TokenIndex *>(bsearch(&tok, sorted_vocab, vocab_size, sizeof(TokenIndex),
                                                            compare_tokens));
        return res != nullptr ? res->id : -1;
    }

    void Tokenizer::encode(const std::string &text, const bool bos, const bool eos, std::vector<int> &tokens,
                           int &n_tokens) {
        if (text.empty()) {
            n_tokens = 0;
            return;
        }

        if (sorted_vocab == nullptr) {
            // lazily malloc and sort the vocabulary
            sorted_vocab = static_cast<TokenIndex *>(malloc(vocab_size * sizeof(TokenIndex)));
            for (int i = 0; i < vocab_size; i++) {
                sorted_vocab[i].str = vocab[i];
                sorted_vocab[i].id = i;
            }
            qsort(sorted_vocab, vocab_size, sizeof(TokenIndex), compare_tokens);
        }

        const auto str_buffer = static_cast<char *>(malloc((max_token_length * 2 + 1 + 2) * sizeof(char)));
        size_t str_len = 0;

        // start at 0 tokens
        n_tokens = 0;

        // Resize output vector to be safe/large enough
        tokens.resize(text.size() + 3); // Upper bound approximation + overhead

        // add optional BOS (=128000) token, if desired
        if (bos)
            tokens[n_tokens++] = 128000;

        // process the raw (UTF-8) byte sequence of the input string
        for (const char *c = text.c_str(); *c != '\0'; c++) {
            if ((*c & 0xC0) != 0x80) {
                str_len = 0;
            }

            str_buffer[str_len++] = *c;
            str_buffer[str_len] = '\0';

            if ((*(c + 1) & 0xC0) == 0x80 && str_len < 4) {
                continue;
            }

            if (const int id = str_lookup(str_buffer, sorted_vocab, vocab_size); id != -1) {
                tokens[n_tokens++] = id;
            } else {
                for (int i = 0; i < str_len; i++) {
                    tokens[n_tokens++] = static_cast<unsigned char>(str_buffer[i]) + 3;
                }
            }
            str_len = 0;
        }

        while (true) {
            float best_score = -1e10;
            int best_id = -1;
            int best_idx = -1;
            int best_len = 2;

            for (int i = 0; i < (n_tokens - 1); i++) {
                snprintf(str_buffer, max_token_length * 2 + 3, "%s%s", vocab[tokens[i]], vocab[tokens[i + 1]]);
                if (const int id = str_lookup(str_buffer, sorted_vocab, vocab_size);
                    id != -1 && vocab_scores[id] > best_score) {
                    best_score = vocab_scores[id];
                    best_id = id;
                    best_idx = i;
                }
            }

            if (best_idx == -1) {
                for (int i = 0; i < (n_tokens - 2); i++) {
                    snprintf(str_buffer, max_token_length * 3 + 4, "%s%s%s", vocab[tokens[i]], vocab[tokens[i + 1]],
                             vocab[tokens[i + 2]]);
                    if (const int id = str_lookup(str_buffer, sorted_vocab, vocab_size);
                        id != -1 && vocab_scores[id] > best_score) {
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

        free(str_buffer);
    }
} // namespace Inference