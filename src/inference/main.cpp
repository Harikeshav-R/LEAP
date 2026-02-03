#include "Transformer.h"
#include "Tokenizer.h"
#include "Sampler.h"
#include "Utils.h"
#include "TcpTransport.h"
#include "UdpTransport.h"
#include "KernelTransport.h"
#include <CLI/CLI.hpp>
#include <iostream>
#include <vector>
#include <string>

using namespace Inference;

void read_stdin(const char *guide, std::string &buffer) {
    std::cout << guide;
    std::getline(std::cin, buffer);
}

void generate(Transformer *transformer, Tokenizer *tokenizer, Sampler *sampler, const std::string &prompt,
              const int steps) {
    std::vector<int> prompt_tokens;
    int num_prompt_tokens = 0;
    tokenizer->encode(prompt, true, false, prompt_tokens, num_prompt_tokens);

    if (num_prompt_tokens < 1) {
        std::cerr << "Error: No prompt tokens encoded" << std::endl;
        std::exit(EXIT_FAILURE);
    }

    long long start = 0;
    int next;
    int token = prompt_tokens[0];
    int pos = 0;

    while (pos < steps) {
        int flags = (pos < num_prompt_tokens - 1) ? FLAG_NO_REPLY : FLAG_NEED_REPLY;
        float *logits = transformer->forward(token, pos, flags);

        if (pos < num_prompt_tokens - 1) {
            next = prompt_tokens[pos + 1];
        } else {
            next = sampler->sample(logits);
        }
        pos++;

        if ((next == 128001 || next == 128009) && pos > num_prompt_tokens) {
            break;
        }

        const std::string &piece = tokenizer->decode(token, next);
        Utils::safe_print(piece);
        std::cout.flush();
        token = next;

        if (start == 0) start = Utils::time_in_ms();
    }
    std::cout << std::endl;

    if (pos > 1) {
        const long long end = Utils::time_in_ms();
        std::cerr << "achieved tok/s: " << (pos - 1) / static_cast<double>(end - start) * 1000 << std::endl;
    }
}

void chat(Transformer *transformer, Tokenizer *tokenizer, Sampler *sampler, const std::string &cli_user_prompt,
          const std::string &cli_system_prompt, const int steps) {
    std::string system_prompt_str;
    std::string user_prompt_str;
    std::vector<int> prompt_tokens;

    prompt_tokens.reserve(steps);

    int user_idx = 0;
    bool user_turn = true;
    int next = 0;
    int token;
    int pos = 0;

    while (pos < steps) {
        if (user_turn) {
            if (pos == 0) {
                prompt_tokens.push_back(128000); // <|begin_of_text|>
                prompt_tokens.push_back(128006); // <|start_header_id|>
                prompt_tokens.push_back(9125); // system
                prompt_tokens.push_back(128007); // <|end_header_id|>
                prompt_tokens.push_back(271); // \n\n
                if (cli_system_prompt.empty()) {
                    read_stdin("Enter system prompt (optional): ", system_prompt_str);
                } else {
                    system_prompt_str = cli_system_prompt;
                }

                if (!system_prompt_str.empty()) {
                    std::vector<int> sys_tokens;
                    int n_sys = 0;
                    tokenizer->encode(system_prompt_str, false, false, sys_tokens, n_sys);
                    prompt_tokens.insert(prompt_tokens.end(), sys_tokens.begin(), sys_tokens.begin() + n_sys);
                }
                prompt_tokens.push_back(128009); // <|eot_id|>
            } else {
                prompt_tokens.clear();
            }

            prompt_tokens.push_back(128006); // <|start_header_id|>
            prompt_tokens.push_back(882); // user
            prompt_tokens.push_back(128007); // <|end_header_id|>
            prompt_tokens.push_back(271); // \n\n
            if (pos == 0 && !cli_user_prompt.empty()) {
                user_prompt_str = cli_user_prompt;
            } else {
                read_stdin("User (or exit): ", user_prompt_str);
                if (user_prompt_str == "exit") break;
            }

            std::vector<int> usr_tokens;
            int n_usr = 0;
            tokenizer->encode(user_prompt_str, false, false, usr_tokens, n_usr);
            prompt_tokens.insert(prompt_tokens.end(), usr_tokens.begin(), usr_tokens.begin() + n_usr);

            prompt_tokens.push_back(128009); // <|eot_id|>
            prompt_tokens.push_back(128006); // <|start_header_id|>
            prompt_tokens.push_back(78191); // assistant
            prompt_tokens.push_back(128007); // <|end_header_id|>
            prompt_tokens.push_back(271); // \n\n
            user_idx = 0;
            user_turn = false;
            std::cout << "Assistant: ";
        }

        if (user_idx < (int)prompt_tokens.size()) {
            token = prompt_tokens[user_idx++];
        } else {
            token = next;
        }

        if (user_idx >= (int)prompt_tokens.size() && (token == 128009 || token == 128001)) {
            user_turn = true;
        }

        int flags = (user_idx < (int)prompt_tokens.size()) ? FLAG_NO_REPLY : FLAG_NEED_REPLY;
        float *logits = transformer->forward(token, pos, flags);
        
        if (flags == FLAG_NO_REPLY) {
            next = (user_idx < (int)prompt_tokens.size()) ? prompt_tokens[user_idx] : 0;
        } else {
            next = sampler->sample(logits);
        }
        
        pos++;

        if (user_idx >= (int)prompt_tokens.size() && (next == 128009 || next == 128001)) {
            std::cout << std::endl;
        }
        if (user_idx >= (int)prompt_tokens.size() && next != 128009 && next != 128001 && next != 128006) {
            const std::string &piece = tokenizer->decode(token, next);
            Utils::safe_print(piece);
            std::cout.flush();
        }
    }
    std::cout << std::endl;
}

int main(int argc, char *argv[]) {
    CLI::App app{"LEAP - Optimized Distributed LLM Inference (Ring Topology)"};
    
    std::string model_path;
    std::string tokenizer_path = "tokenizer.bin";
    app.add_option("model", model_path, "Path to model (.bin)")->required()->check(CLI::ExistingFile);

    std::string role = "single";
    app.add_option("--role", role, "Node role: single, master, worker")->capture_default_str();
    
    std::string transport_type = "tcp";
    app.add_option("--transport", transport_type, "Transport: tcp, udp, kernel")->capture_default_str();

    int port = 9999;
    app.add_option("--port", port, "Local port to bind")->capture_default_str();

    std::string next_host;
    app.add_option("--next-host", next_host, "Next node IP");

    int next_port = 9999;
    app.add_option("--next-port", next_port, "Next node port")->capture_default_str();

    int split = 0;
    app.add_option("--split", split, "Start layer index")->capture_default_str();

    int end = 0;
    app.add_option("--end", end, "End layer index (exclusive, 0=all)")->capture_default_str();

    std::string prompt;
    app.add_option("-p,--prompt", prompt, "Input prompt");

    bool chat_mode = false;
    app.add_flag("-c,--chat", chat_mode, "Chat mode");

    int n_predict = 4096;
    app.add_option("-n,--n-predict", n_predict, "Max tokens")->capture_default_str();

    float temperature = 1.0f;
    app.add_option("--temp", temperature, "Sampling temperature")->capture_default_str();

    CLI11_PARSE(app, argc, argv);

    try {
        auto transformer = Transformer::create(model_path);
        if (end == 0) end = transformer->config.n_layers;

        DistributedMode dist_role = DistributedMode::Single;
        if (role == "master") dist_role = DistributedMode::Master;
        else if (role == "worker") dist_role = DistributedMode::Worker;

        std::unique_ptr<Transport> transport = nullptr;
        if (dist_role != DistributedMode::Single) {
            if (next_host.empty()) throw std::runtime_error("--next-host is required for distributed mode");

            if (transport_type == "tcp") {
                transport = std::make_unique<TcpTransport>("0.0.0.0", port, next_host, next_port);
            } else if (transport_type == "udp") {
                transport = std::make_unique<UdpTransport>("0.0.0.0", port, next_host, next_port);
            } else if (transport_type == "kernel") {
#ifndef __linux__
                throw std::runtime_error("Kernel transport is only supported on Linux.");
#else
                transport = std::make_unique<KernelTransport>(port, next_host, next_port);
#endif
            }
            if (transport) transport->initialize();
        }

        DistributedConfig dist_config;
        dist_config.mode = dist_role;
        dist_config.split_layer = split;
        dist_config.end_layer = end;
        dist_config.transport = transport.get();
        
        transformer->set_distributed_config(dist_config);

        if (dist_role == DistributedMode::Worker) {
            transformer->worker_loop();
            return 0;
        }

        Tokenizer tokenizer(tokenizer_path, transformer->config.vocab_size);
        Sampler sampler(transformer->config.vocab_size, temperature, 0.9f, 0);

        if (chat_mode) chat(transformer.get(), &tokenizer, &sampler, prompt, "", n_predict);
        else generate(transformer.get(), &tokenizer, &sampler, prompt, n_predict);

    } catch (const std::exception &e) {
        std::cerr << "[Error] " << e.what() << std::endl;
        return 1;
    }
    return 0;
}
