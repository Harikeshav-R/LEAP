#include "Transformer.h"
#include "Tokenizer.h"
#include "Sampler.h"
#include "Utils.h"
#include "TcpTransport.h"
#include "UdpTransport.h"
#include "KernelTransport.h"
#include <iostream>
#include <vector>
#include <string>

using namespace Inference;

void read_stdin(const char *guide, std::string &buffer) {
    // ... (rest of file)
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

        float *logits = transformer->forward(token, pos);
        next = sampler->sample(logits);
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

void error_usage() {
    std::cerr << "Usage:   run <checkpoint> [options]\n";
    std::cerr << "Example: run model.bin -n 4096 -i \"Once upon a time\"\n";
    std::cerr << "Options:\n";
    std::cerr << "  -t <float>  temperature in [0,inf], default 1.0\n";
    std::cerr << "  -p <float>  p value in top-p (nucleus) sampling in [0,1] default 0.9\n";
    std::cerr << "  -s <int>    random seed, default time(NULL)\n";
    std::cerr << "  -n <int>    number of steps to run for, default 4096. 0 = max_seq_len\n";
    std::cerr << "  -i <string> input prompt\n";
    std::cerr << "  -z <string> optional path to custom tokenizer\n";
    std::cerr << "  -m <string> mode: generate|chat, default: generate\n";
    std::cerr << "  -y <string> (optional) system prompt in chat mode\n";
    std::cerr << "Distributed Options:\n";
    std::cerr << "  --dist <mode>  distributed mode: single|master|worker|master-udp|worker-udp|worker-kernel, default: single\n";
    std::cerr << "  --ip <string>  worker IP address (for master) or bind address (for worker), default: 0.0.0.0\n";
    std::cerr << "  --master-ip <string> master IP address (only for worker-kernel mode)\n";
    std::cerr << "  --port <int>   port number, default: 9999\n";
    std::cerr << "  --split <int>  layer index to split at, default: 0\n";
    std::exit(EXIT_FAILURE);
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

    // Distributed args
    std::string dist_mode_str = "single";
    std::string ip = "0.0.0.0";
    std::string master_ip = "";
    int port = 9999;
    std::string next_host = "";
    int next_port = 0;
    int split_layer = 0;

    if (argc >= 2) {
        checkpoint_path = argv[1];
    } else {
        error_usage();
    }

    for (int i = 2; i < argc; i += 2) {
        if (i + 1 >= argc) error_usage();
        // Check for long args
        std::string arg = argv[i];
        if (arg == "--dist") dist_mode_str = argv[i + 1];
        else if (arg == "--ip") ip = argv[i + 1];
        else if (arg == "--master-ip") master_ip = argv[i + 1];
        else if (arg == "--port") port = std::stoi(argv[i + 1]);
        else if (arg == "--next-host") next_host = argv[i + 1];
        else if (arg == "--next-port") next_port = std::stoi(argv[i + 1]);
        else if (arg == "--split") split_layer = std::stoi(argv[i + 1]);
        else if (arg[0] == '-') {
            if (arg.length() != 2) error_usage();
            if (argv[i][1] == 't') temperature = std::stof(argv[i + 1]);
            else if (argv[i][1] == 'p') topp = std::stof(argv[i + 1]);
            else if (argv[i][1] == 's') rng_seed = std::stoll(argv[i + 1]);
            else if (argv[i][1] == 'n') steps = std::stoi(argv[i + 1]);
            else if (argv[i][1] == 'i') prompt = argv[i + 1];
            else if (argv[i][1] == 'z') tokenizer_path = argv[i + 1];
            else if (argv[i][1] == 'm') mode = argv[i + 1];
            else if (argv[i][1] == 'y') system_prompt = argv[i + 1];
            else error_usage();
        } else {
            error_usage();
        }
    }

    if (next_port == 0) next_port = port;

    if (rng_seed <= 0) rng_seed = static_cast<unsigned int>(std::time(nullptr));
    if (temperature < 0.0) temperature = 0.0;
    if (topp < 0.0 || 1.0 < topp) topp = 0.9;
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

        // Setup Distributed Mode
        DistributedMode dist_mode = DistributedMode::Single;
        std::unique_ptr<Transport> transport = nullptr;

        if (dist_mode_str == "master") {
            dist_mode = DistributedMode::Master;
            if (split_layer <= 0 || split_layer >= transformer->config.n_layers) {
                std::cerr << "Invalid split layer for master mode" << std::endl;
                return 1;
            }
            transport = std::make_unique<TcpTransport>(ip, port, false, next_host, next_port);
            transport->initialize();
        } else if (dist_mode_str == "worker") {
            dist_mode = DistributedMode::Worker;
            if (split_layer <= 0 || split_layer >= transformer->config.n_layers) {
                std::cerr << "Invalid split layer for worker mode" << std::endl;
                return 1;
            }
            transport = std::make_unique<TcpTransport>(ip, port, true, next_host, next_port);
            transport->initialize();
        } else if (dist_mode_str == "udp") {
            // UDP Mode (Standard User Space UDP)
            // Use --dist udp for both Master (client) and Worker (server) logic
            // But we need to know role. Let's reuse 'master/worker' logic but add transport type arg?
            // Simpler: --dist master-udp / --dist worker-udp / --dist worker-kernel
            std::cerr << "Use specific modes: master-udp, worker-udp, worker-kernel" << std::endl;
            return 1;
        } else if (dist_mode_str == "master-udp") {
            dist_mode = DistributedMode::Master;
            transport = std::make_unique<UdpTransport>(ip, port, false, next_host, next_port);
            transport->initialize();
        } else if (dist_mode_str == "worker-udp") {
            dist_mode = DistributedMode::Worker;
            transport = std::make_unique<UdpTransport>(ip, port, true, next_host, next_port);
            transport->initialize();
        } else if (dist_mode_str == "worker-kernel") {
#ifndef __linux__
            std::cerr << "Error: --dist worker-kernel is only supported on Linux." << std::endl;
            return 1;
#else
            dist_mode = DistributedMode::Worker;
            // For worker-kernel, we need the Master's IP to set as destination
            // If next_host is provided (Ring Mode), use that. Otherwise use master_ip (Legacy).
            std::string target_ip = !next_host.empty() ? next_host : (master_ip.empty() ? ip : master_ip);
            transport = std::make_unique<KernelTransport>(target_ip, port);
            transport->initialize();
#endif
        } else if (dist_mode_str != "single") {
            std::cerr << "Unknown distributed mode: " << dist_mode_str << std::endl;
            return 1;
        }

        transformer->set_distributed_config(dist_mode, split_layer, transport.get());

        if (dist_mode == DistributedMode::Worker) {
            // Worker enters loop and stays there
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
        } else {
            std::cerr << "unknown mode: " << mode << std::endl;
            error_usage();
        }
    } catch (const std::exception &e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}