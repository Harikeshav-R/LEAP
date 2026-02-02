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

// ... (Keep existing helper functions like read_stdin, generate, chat) ...
void read_stdin(const char *guide, std::string &buffer) {
    std::cout << guide;
    std::getline(std::cin, buffer);
}

void generate(Transformer *transformer, Tokenizer *tokenizer, Sampler *sampler, const std::string &prompt,
              const int steps) {
    const std::string &actual_prompt = prompt;

    std::vector<int> prompt_tokens;
    int num_prompt_tokens = 0;
    tokenizer->encode(actual_prompt, true, false, prompt_tokens, num_prompt_tokens);

    if (num_prompt_tokens < 1) {
        std::cerr << "something is wrong, expected at least 1 prompt token" << std::endl;
        std::exit(EXIT_FAILURE);
    }

    long long start = 0;
    int next;
    int token = prompt_tokens[0];
    int pos = 0;

    while (pos < steps) {
        // Pipelining: Fire-and-forget for prompt tokens except the last one
        int flags = (pos < num_prompt_tokens - 1) ? FLAG_NO_REPLY : FLAG_NEED_REPLY;
        float *logits = transformer->forward(token, pos, flags);

        if (pos < num_prompt_tokens - 1) {
            next = prompt_tokens[pos + 1];
        } else {
            // We only need logits for the last prompt token and generation
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

        if (start == 0) {
            start = Utils::time_in_ms();
        }
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
                // Prepend system prompt
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

        if (user_idx < prompt_tokens.size()) {
            token = prompt_tokens[user_idx++];
        } else {
            token = next;
        }

        if (user_idx >= prompt_tokens.size() && (token == 128009 || token == 128001)) {
            user_turn = true;
        }

        int flags = FLAG_NEED_REPLY;
        if (user_idx < prompt_tokens.size() - 1) {
            flags = FLAG_NO_REPLY;
        }
        
        if (user_idx < prompt_tokens.size()) {
            flags = FLAG_NO_REPLY;
        }

        float *logits = transformer->forward(token, pos, flags);
        
        if (flags == FLAG_NO_REPLY) {
            next = 0; // Dummy
        } else {
            next = sampler->sample(logits);
        }
        
        pos++;

        if (user_idx >= prompt_tokens.size() && (next == 128009 || next == 128001)) {
            std::cout << std::endl;
        }
        if (user_idx >= prompt_tokens.size() && next != 128009 && next != 128001 && next != 128006) {
            const std::string &piece = tokenizer->decode(token, next);
            Utils::safe_print(piece);
            std::cout.flush();
        }
    }
    std::cout << std::endl;
}

int main(int argc, char *argv[]) {
    CLI::App app{"LEAP Inference Engine\nA high-performance, distributed LLM inference runner.\n"};
    argv = app.ensure_utf8(argv);

    std::string checkpoint_path;
    std::string tokenizer_path = "tokenizer.bin";
    float temperature = 1.0f;
    float topp = 0.9f;
    int steps = 4096;
    std::string prompt;
    unsigned long long rng_seed = 0;
    std::string mode = "generate";
    std::string system_prompt;

    // Distributed args
    std::string dist_mode_str = "single";
    std::string ip = "0.0.0.0";
    std::string master_ip = "";
    int port = 9999;
    int split_layer = 0;
    std::string next_ip = "";
    int next_port = 0;
    int end_layer = 0;

    // --- Options ---
    app.add_option("checkpoint", checkpoint_path, "Path to the model checkpoint file (e.g., model.bin)")
        ->required()
        ->check(CLI::ExistingFile);

    // Group: General
    auto *gen_opts = app.add_option_group("General Configuration");
    gen_opts->add_option("-z,--tokenizer", tokenizer_path, "Path to the tokenizer file")
            ->check(CLI::ExistingFile);
    gen_opts->add_option("-m,--mode", mode, "Inference mode: 'generate' for completion, 'chat' for dialog")
            ->check(CLI::IsMember({"generate", "chat"}));
    gen_opts->add_option("-n,--steps", steps, "Maximum number of steps to run (0 = max_seq_len)");
    gen_opts->add_option("-s,--seed", rng_seed, "Random seed (0 = use time)");

    // Group: Sampling
    auto *sample_opts = app.add_option_group("Sampling Parameters");
    sample_opts->add_option("-t,--temperature", temperature, "Temperature for sampling [0.0, inf)")
               ->check(CLI::NonNegativeNumber);
    sample_opts->add_option("-p,--top-p", topp, "Top-P (Nucleus) sampling probability [0.0, 1.0]")
               ->check(CLI::Range(0.0, 1.0));

    // Group: Input
    auto *input_opts = app.add_option_group("Input");
    input_opts->add_option("-i,--prompt", prompt, "Initial user prompt");
    input_opts->add_option("-y,--system-prompt", system_prompt, "System prompt (only used in chat mode)");

    // Group: Distributed
    auto *dist_opts = app.add_option_group("Distributed Inference", "Configuration for multi-node inference");
    dist_opts->add_option("--dist", dist_mode_str, "Distributed mode")
             ->check(CLI::IsMember({"single", "master", "worker", "master-udp", "worker-udp", "worker-kernel"}));
    
    dist_opts->add_option("--ip", ip, "Bind IP address (Worker only). Default: 0.0.0.0");
    dist_opts->add_option("--port", port, "Bind Port (Worker only). Default: 9999")
             ->check(CLI::Range(1, 65535));
    dist_opts->add_option("--split", split_layer, "Layer index to split at (start layer for worker)");
    dist_opts->add_option("--end-layer", end_layer, "Layer index to stop at (exclusive, for worker). Default: n_layers");
    
    dist_opts->add_option("--master-ip", master_ip, "Master IP address (required for worker-kernel mode)");
    
    dist_opts->add_option("--next-ip", next_ip, "Target IP address for outgoing connection (Master -> Worker 1, Worker -> Next)");
    dist_opts->add_option("--next-port", next_port, "Target Port for outgoing connection")
             ->check(CLI::Range(0, 65535)); // 0 means unset

    CLI11_PARSE(app, argc, argv);

    // --- Post-Processing / Defaults ---
    if (rng_seed <= 0) rng_seed = static_cast<unsigned int>(std::time(nullptr));
    if (steps < 0) steps = 0;

    try {
        auto transformer = Transformer::create(checkpoint_path);

        std::cout << "Model loaded successfully." << std::endl;
        std::cout << "Config: ["
                << transformer->config.n_layers << " layers, "
                << transformer->config.dim << " dim, "
                << transformer->config.n_heads << " heads, "
                << transformer->config.vocab_size << " vocab, "
                << transformer->config.seq_len << " seq_len]" << std::endl;

        if (steps == 0 || steps > transformer->config.seq_len) {
            steps = transformer->config.seq_len;
        }
        
        if (end_layer == 0) end_layer = transformer->config.n_layers;

        // Setup Distributed Mode
        DistributedMode dist_mode = DistributedMode::Single;
        std::unique_ptr<Transport> transport = nullptr;

        if (dist_mode_str == "master") {
            dist_mode = DistributedMode::Master;
            if (split_layer <= 0 || split_layer >= transformer->config.n_layers) {
                std::cerr << "Error: Invalid split layer for master mode. Must be > 0 and < n_layers." << std::endl;
                return 1;
            }
            if (next_ip.empty() || next_port == 0) {
                std::cerr << "Error: Master mode requires --next-ip and --next-port to connect to the first worker." << std::endl;
                return 1;
            }
            
            transport = std::make_unique<TcpTransport>(next_ip, next_port, false, "", 0);
            transport->initialize();
        } else if (dist_mode_str == "worker") {
            dist_mode = DistributedMode::Worker;
            if (split_layer <= 0 || split_layer >= transformer->config.n_layers) {
                std::cerr << "Error: Invalid split layer for worker mode." << std::endl;
                return 1;
            }
            transport = std::make_unique<TcpTransport>(ip, port, true, next_ip, next_port);
            transport->initialize();
        } else if (dist_mode_str == "udp") {
             std::cerr << "Error: Use specific modes: master-udp, worker-udp, worker-kernel" << std::endl;
             return 1;
        } else if (dist_mode_str == "master-udp") {
            dist_mode = DistributedMode::Master;
            if (next_ip.empty() || next_port == 0) {
                std::cerr << "Error: Master UDP mode requires --next-ip and --next-port." << std::endl;
                return 1;
            }
            transport = std::make_unique<UdpTransport>(next_ip, next_port, false);
            transport->initialize();
        } else if (dist_mode_str == "worker-udp") {
            dist_mode = DistributedMode::Worker;
            transport = std::make_unique<UdpTransport>(ip, port, true, next_ip, next_port);
            transport->initialize();
        } else if (dist_mode_str == "worker-kernel") {
#ifndef __linux__
            std::cerr << "Error: --dist worker-kernel is only supported on Linux." << std::endl;
            return 1;
#else
            dist_mode = DistributedMode::Worker;
            std::string target_ip = master_ip.empty() ? ip : master_ip;
            transport = std::make_unique<KernelTransport>(target_ip, port);
            transport->initialize();
#endif
        } else if (dist_mode_str != "single") {
             // CLI11 validation covers this, but safe fallback
             std::cerr << "Error: Unknown distributed mode: " << dist_mode_str << std::endl;
             return 1;
        }

        DistributedConfig dist_config;
        dist_config.mode = dist_mode;
        dist_config.split_layer = split_layer;
        dist_config.end_layer = end_layer;
        dist_config.transport = transport.get();
        dist_config.next_ip = next_ip;
        dist_config.next_port = next_port;
        dist_config.is_tail = (dist_mode == DistributedMode::Worker && next_ip.empty());
        
        transformer->set_distributed_config(dist_config);

        if (dist_mode == DistributedMode::Worker) {
            transformer->worker_loop();
            return 0;
        }

        // Master or Single mode proceeds
        Tokenizer tokenizer(tokenizer_path, transformer->config.vocab_size);
        Sampler sampler(transformer->config.vocab_size, temperature, topp, rng_seed);

        if (mode == "generate") {
            generate(transformer.get(), &tokenizer, &sampler, prompt, steps);
        } else if (mode == "chat") {
            chat(transformer.get(), &tokenizer, &sampler, prompt, system_prompt, steps);
        }
    } catch (const std::exception &e) {
        std::cerr << "Runtime Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}