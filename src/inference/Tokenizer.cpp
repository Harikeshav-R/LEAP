#include "Tokenizer.h"
#include <stdexcept>
#include <iostream>
#include <fstream>
#include <algorithm>
#include <vector>
#include <queue>
#include <tuple>

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
        vocab_lookup.reserve(vocab_size);

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
            vocab_lookup[vocab[i]] = i;
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

    int Tokenizer::str_lookup(std::string_view str) {
        const auto it = vocab_lookup.find(str);
        if (it != vocab_lookup.end()) {
            return it->second;
        }
        return -1;
    }

    void Tokenizer::encode(const std::string &text, const bool bos, const bool eos, std::vector<int> &tokens,
                           int &n_tokens) {
        if (text.empty()) {
            n_tokens = 0;
            return;
        }

        std::string str_buffer;
        str_buffer.reserve(max_token_length * 2 + 10);

        n_tokens = 0;
        tokens.clear();
        tokens.reserve(text.size() + 3);

        if (bos)
            tokens.push_back(128000);

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

            if (const int id = str_lookup(str_buffer); id != -1) {
                tokens.push_back(id);
            } else {
                for (const char b: str_buffer) {
                    tokens.push_back(static_cast<unsigned char>(b) + 3);
                }
            }
            str_buffer.clear();
        }

        n_tokens = tokens.size();

        // Linked list structure for O(1) removals
        struct Node {
            int id;
            int prev;
            int next;
        };
        std::vector<Node> list(n_tokens);
        for (int i = 0; i < n_tokens; ++i) {
            list[i].id = tokens[i];
            list[i].prev = i - 1;
            list[i].next = i + 1;
        }
        list.back().next = -1; // End marker

        // Buffer for merging strings to avoid allocations
        std::vector<char> merge_buffer;
        merge_buffer.reserve(max_token_length * 2 + 1);

        // Priority Queue for O(N log N) merge
        // Tuple: <score, left_idx, right_idx, left_token_id, right_token_id>
        // We store token IDs to validate that the pair is still valid when popped
        using PqElement = std::tuple<float, int, int, int, int>;

        auto comp = [](const PqElement &a, const PqElement &b) {
            // Priority 1: Higher score
            if (std::abs(std::get < 0 > (a) - std::get < 0 > (b)) > 1e-6) {
                return std::get < 0 > (a) < std::get < 0 > (b);
            }
            // Priority 2: Smaller index (leftmost first)
            return std::get < 1 > (a) > std::get < 1 > (b);
        };

        std::priority_queue<PqElement, std::vector<PqElement>, decltype(comp)> queue(comp);

        // Initial scan to fill PQ
        for (int i = 0; i != -1 && list[i].next != -1; i = list[i].next) {
            const int next_i = list[i].next;
            const std::string &s1 = vocab[list[i].id];
            const std::string &s2 = vocab[list[next_i].id];

            merge_buffer.clear();
            merge_buffer.insert(merge_buffer.end(), s1.begin(), s1.end());
            merge_buffer.insert(merge_buffer.end(), s2.begin(), s2.end());
            std::string_view merged_view(merge_buffer.data(), merge_buffer.size());

            if (const int id = str_lookup(merged_view); id != -1) {
                queue.emplace(vocab_scores[id], i, next_i, list[i].id, list[next_i].id);
            }
        }

        // BPE merge loop
        while (!queue.empty()) {
            const auto [score, left_idx, right_idx, left_id, right_id] = queue.top();
            queue.pop();

            // Validate if the pair is still current
            // 1. Check if nodes are still adjacent
            if (list[left_idx].next != right_idx) continue;
            // 2. Check if token IDs match (nodes haven't been modified)
            if (list[left_idx].id != left_id || list[right_idx].id != right_id) continue;

            // Perform Merge
            const std::string &s1 = vocab[left_id];
            const std::string &s2 = vocab[right_id];

            merge_buffer.clear();
            merge_buffer.insert(merge_buffer.end(), s1.begin(), s1.end());
            merge_buffer.insert(merge_buffer.end(), s2.begin(), s2.end());
            std::string_view merged_view(merge_buffer.data(), merge_buffer.size());

            const int merged_id = str_lookup(merged_view);
            if (merged_id == -1) continue; // Should not happen given initial scan, but safety first

            // Update list
            list[left_idx].id = merged_id;

            // Remove right_idx
            int next_next_i = list[right_idx].next;
            list[left_idx].next = next_next_i;
            if (next_next_i != -1) {
                list[next_next_i].prev = left_idx;
            }

            // Add new potential pairs to PQ

            // 1. New pair with left neighbor
            if (int prev_i = list[left_idx].prev; prev_i != -1) {
                const std::string &ps1 = vocab[list[prev_i].id];
                const std::string &ps2 = vocab[merged_id];

                merge_buffer.clear();
                merge_buffer.insert(merge_buffer.end(), ps1.begin(), ps1.end());
                merge_buffer.insert(merge_buffer.end(), ps2.begin(), ps2.end());
                std::string_view p_view(merge_buffer.data(), merge_buffer.size());

                if (const int pid = str_lookup(p_view); pid != -1) {
                    queue.emplace(vocab_scores[pid], prev_i, left_idx, list[prev_i].id, merged_id);
                }
            }

            // 2. New pair with right neighbor
            if (next_next_i != -1) {
                const std::string &ns1 = vocab[merged_id];
                const std::string &ns2 = vocab[list[next_next_i].id];

                merge_buffer.clear();
                merge_buffer.insert(merge_buffer.end(), ns1.begin(), ns1.end());
                merge_buffer.insert(merge_buffer.end(), ns2.begin(), ns2.end());
                std::string_view n_view(merge_buffer.data(), merge_buffer.size());

                if (const int nid = str_lookup(n_view); nid != -1) {
                    queue.emplace(vocab_scores[nid], left_idx, next_next_i, merged_id, list[next_next_i].id);
                }
            }
        }

        // Reconstruct tokens vector
        tokens.clear();
        for (int i = 0; i != -1; i = list[i].next) {
            tokens.push_back(list[i].id);
        }
        n_tokens = tokens.size();

        if (eos) {
            tokens.push_back(128001);
            n_tokens++;
        }
    }
} // namespace Inference