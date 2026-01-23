#include "Transformer.h"
#include "Tokenizer.h"
#include "Sampler.h"
#include "Utils.h"
#include <iostream>
#include <vector>
#include <string>
#include <unistd.h>

using namespace Inference;

void read_stdin(const char *guide, char *buffer, size_t bufsize) {
    printf("%s", guide);
    if (fgets(buffer, bufsize, stdin) != nullptr) {
        if (const size_t len = strlen(buffer); len > 0 && buffer[len - 1] == '\n') {
            buffer[len - 1] = '\0';
        }
    }
}

void generate(Transformer *transformer, Tokenizer *tokenizer, Sampler *sampler, const std::string &prompt,
              const int steps) {
    const std::string &actual_prompt = prompt;

    std::vector<int> prompt_tokens;
    int num_prompt_tokens = 0;
    tokenizer->encode(actual_prompt, true, false, prompt_tokens, num_prompt_tokens);

    if (num_prompt_tokens < 1) {
        fprintf(stderr, "something is wrong, expected at least 1 prompt token\n");
        exit(EXIT_FAILURE);
    }

    long long start = 0;
    int next;
    int token = prompt_tokens[0];
    int pos = 0;

    while (pos < steps) {
        float *logits = transformer->forward(token, pos);

        if (pos < num_prompt_tokens - 1) {
            next = prompt_tokens[pos + 1];
        } else {
            next = sampler->sample(logits);
        }
        pos++;

        if ((next == 128001 || next == 128009) && pos > num_prompt_tokens) {
            break;
        }

        const char *piece = tokenizer->decode(token, next);
        Utils::safe_printf(piece);
        fflush(stdout);
        token = next;

        if (start == 0) {
            start = Utils::time_in_ms();
        }
    }
    printf("\n");

    if (pos > 1) {
        const long long end = Utils::time_in_ms();
        fprintf(stderr, "achieved tok/s: %f\n", (pos - 1) / static_cast<double>(end - start) * 1000);
    }
}

void chat(Transformer *transformer, Tokenizer *tokenizer, Sampler *sampler, const std::string &cli_user_prompt,
          const std::string &cli_system_prompt, const int steps) {
    char system_prompt[32768];
    char user_prompt[32768];
    std::vector<int> prompt_tokens;

    // We'll use a dynamic vector for prompt tokens, but C implementation used a fixed large buffer.
    // We should be careful about resizing.
    prompt_tokens.reserve(steps); // Reserve enough space

    int user_idx = 0;
    bool user_turn = true;
    int next = 0;
    int token;
    int pos = 0;

    while (pos < steps) {
        if (user_turn) {
            if (pos == 0) {
                // Prepend system prompt
                prompt_tokens.push_back(128000); // <|begin_of_text|>
                prompt_tokens.push_back(128006); // <|start_header_id|>
                prompt_tokens.push_back(9125); // system
                prompt_tokens.push_back(128007); // <|end_header_id|>
                prompt_tokens.push_back(271); // \n\n
                if (cli_system_prompt.empty()) {
                    read_stdin("Enter system prompt (optional): ", system_prompt, sizeof(system_prompt));
                } else {
                    strncpy(system_prompt, cli_system_prompt.c_str(), sizeof(system_prompt));
                }

                if (strlen(system_prompt) > 0) {
                    std::vector<int> sys_tokens;
                    int n_sys = 0;
                    tokenizer->encode(system_prompt, false, false, sys_tokens, n_sys);
                    prompt_tokens.insert(prompt_tokens.end(), sys_tokens.begin(), sys_tokens.begin() + n_sys);
                }
                prompt_tokens.push_back(128009); // <|eot_id|>
            } else {
                // Clear previous turn tokens if needed, but we append to `prompt_tokens` which acts as the queue for current turn
                // Actually, `prompt_tokens` in C implementation accumulates ONLY the current turn inputs?
                // No, it seems to just reset `num_prompt_tokens = 0` if pos != 0.
                // So `prompt_tokens` holds the tokens to be fed *now*.
                prompt_tokens.clear();
            }

            prompt_tokens.push_back(128006); // <|start_header_id|>
            prompt_tokens.push_back(882); // user
            prompt_tokens.push_back(128007); // <|end_header_id|>
            prompt_tokens.push_back(271); // \n\n

            if (pos == 0 && !cli_user_prompt.empty()) {
                strncpy(user_prompt, cli_user_prompt.c_str(), sizeof(user_prompt));
            } else {
                read_stdin("User (or exit): ", user_prompt, sizeof(user_prompt));
                if (strcmp(user_prompt, "exit") == 0) break;
            }

            std::vector<int> usr_tokens;
            int n_usr = 0;
            tokenizer->encode(user_prompt, false, false, usr_tokens, n_usr);
            prompt_tokens.insert(prompt_tokens.end(), usr_tokens.begin(), usr_tokens.begin() + n_usr);

            prompt_tokens.push_back(128009); // <|eot_id|>
            prompt_tokens.push_back(128006); // <|start_header_id|>
            prompt_tokens.push_back(78191); // assistant
            prompt_tokens.push_back(128007); // <|end_header_id|>
            prompt_tokens.push_back(271); // \n\n

            user_idx = 0;
            user_turn = false;
            printf("Assistant: ");
        }

        if (user_idx < prompt_tokens.size()) {
            token = prompt_tokens[user_idx++];
        } else {
            token = next;
        }

        if (user_idx >= prompt_tokens.size() && (token == 128009 || token == 128001)) {
            user_turn = true;
        }

        float *logits = transformer->forward(token, pos);
        next = sampler->sample(logits);
        pos++;

        if (user_idx >= prompt_tokens.size() && (next == 128009 || next == 128001)) {
            printf("\n");
        }
        if (user_idx >= prompt_tokens.size() && next != 128009 && next != 128001 && next != 128006) {
            const char *piece = tokenizer->decode(token, next);
            Utils::safe_printf(piece);
            fflush(stdout);
        }
    }
    printf("\n");
}

void error_usage() {
    fprintf(stderr, "Usage:   run <checkpoint> [options]\n");
    fprintf(stderr, "Example: run model.bin -n 4096 -i \"Once upon a time\"\n");
    fprintf(stderr, "Options:\n");
    fprintf(stderr, "  -t <float>  temperature in [0,inf], default 1.0\n");
    fprintf(stderr, "  -p <float>  p value in top-p (nucleus) sampling in [0,1] default 0.9\n");
    fprintf(stderr, "  -s <int>    random seed, default time(NULL)\n");
    fprintf(stderr, "  -n <int>    number of steps to run for, default 4096. 0 = max_seq_len\n");
    fprintf(stderr, "  -i <string> input prompt\n");
    fprintf(stderr, "  -z <string> optional path to custom tokenizer\n");
    fprintf(stderr, "  -m <string> mode: generate|chat, default: generate\n");
    fprintf(stderr, "  -y <string> (optional) system prompt in chat mode\n");
    exit(EXIT_FAILURE);
}

int main(int argc, char *argv[]) {
    std::string checkpoint_path;
    std::string tokenizer_path = "tokenizer.bin";
    float temperature = 1.0f;
    float topp = 0.9f;
    int steps = 4096;
    std::string prompt;
    unsigned long long rng_seed = 0;
    std::string mode = "generate";
    std::string system_prompt;

    if (argc >= 2) {
        checkpoint_path = argv[1];
    } else {
        error_usage();
    }

    for (int i = 2; i < argc; i += 2) {
        if (i + 1 >= argc) error_usage();
        if (argv[i][0] != '-') error_usage();
        if (strlen(argv[i]) != 2) error_usage();

        if (argv[i][1] == 't') temperature = strtof(argv[i + 1], nullptr);
        else if (argv[i][1] == 'p') topp = strtof(argv[i + 1], nullptr);
        else if (argv[i][1] == 's') rng_seed = strtol(argv[i + 1], nullptr, 10);
        else if (argv[i][1] == 'n') steps = strtol(argv[i + 1], nullptr, 10);
        else if (argv[i][1] == 'i') prompt = argv[i + 1];
        else if (argv[i][1] == 'z') tokenizer_path = argv[i + 1];
        else if (argv[i][1] == 'm') mode = argv[i + 1];
        else if (argv[i][1] == 'y') system_prompt = argv[i + 1];
        else error_usage();
    }

    if (rng_seed <= 0) rng_seed = static_cast<unsigned int>(time(nullptr));
    if (temperature < 0.0) temperature = 0.0;
    if (topp < 0.0 || 1.0 < topp) topp = 0.9;
    if (steps < 0) steps = 0;

    try {
        auto transformer = Transformer::create(checkpoint_path);

        if (steps == 0 || steps > transformer->config.seq_len) {
            steps = transformer->config.seq_len;
        }

        Tokenizer tokenizer(tokenizer_path, transformer->config.vocab_size);
        Sampler sampler(transformer->config.vocab_size, temperature, topp, rng_seed);

        if (mode == "generate") {
            generate(transformer.get(), &tokenizer, &sampler, prompt, steps);
        } else if (mode == "chat") {
            chat(transformer.get(), &tokenizer, &sampler, prompt, system_prompt, steps);
        } else {
            fprintf(stderr, "unknown mode: %s\n", mode.c_str());
            error_usage();
        }
    } catch (const std::exception &e) {
        fprintf(stderr, "Error: %s\n", e.what());
        return 1;
    }

    return 0;
}