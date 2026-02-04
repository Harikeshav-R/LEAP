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
        const int flags = (pos < num_prompt_tokens - 1) ? FLAG_NO_REPLY : FLAG_NEED_REPLY;
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
    std::vector<int> prompt_tokens; // Current turn tokens or history for re-eval
    std::vector<int> history_tokens; // Full conversation history

    prompt_tokens.reserve(steps);

    int user_idx = 0;
    bool user_turn = true;
    int next = 0;
    int token;
    int pos = 0;
    bool recomputing = false; // Flag to suppress output during re-eval

    while (pos < steps) {
        if (user_turn && !recomputing) {
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
            bool command_handled = false;

            if (pos == 0 && !cli_user_prompt.empty()) {
                user_prompt_str = cli_user_prompt;
            } else {
                while (true) {
                    read_stdin("User (or exit): ", user_prompt_str);
                    if (user_prompt_str == "exit") break;
                    
                    if (user_prompt_str.rfind("/layer", 0) == 0) {
                        std::istringstream iss(user_prompt_str);
                        std::string cmd;
                        iss >> cmd; 
                        std::vector<int> splits;
                        int val;
                        while (iss >> val) {
                            splits.push_back(val);
                        }
                        
                        if (splits.empty()) {
                            std::cout << "Usage: /layer <end_layer_node0> <end_layer_node1> ... (e.g., /layer 16 32)" << std::endl;
                        } else {
                            std::vector<LayerConfig> configs;
                            int start = 0;
                            for (int end : splits) {
                                configs.push_back({start, end});
                                start = end;
                            }
                            transformer->distribute_config(configs);
                            
                            // --- REWIND LOGIC (BATCHED) ---
                            std::cout << "[System] Re-evaluating context (" << history_tokens.size() << " tokens)..." << std::endl;
                            if (!history_tokens.empty()) {
                                transformer->forward_batch(history_tokens, 0);
                            }
                            
                            // Reset state for continuation
                            prompt_tokens = history_tokens; 
                            pos = history_tokens.size(); // Set pos to end of history
                            user_idx = prompt_tokens.size(); // Skip user input processing
                            
                            // We are done recomputing in one shot.
                            // Resume generation immediately.
                            recomputing = false; 
                            user_turn = false; // Bypass user input
                            
                            command_handled = true;
                            break; 
                        }
                        continue; 
                    } else if (user_prompt_str == "/stats") {
                        std::vector<NodeStats> stats = transformer->collect_stats();
                        std::cout << "\n--- Cluster Statistics ---\n";
                        std::cout << "Node | CPU(%) | RAM(%) | Temp(C) | Layers\n";
                        std::cout << "-----|--------|--------|---------|-------\n";
                        for (size_t i = 0; i < stats.size(); i++) {
                            std::printf("  %zu  |  %5.1f |  %5.1f |  %5.1f  | %d-%d\n", 
                                        i, stats[i].cpu_usage, stats[i].ram_usage, stats[i].temperature,
                                        stats[i].split_layer, stats[i].end_layer);
                        }
                        std::cout << "--------------------------\n\n";
                        continue;
                    }
                    break;
                }
                if (user_prompt_str == "exit") break;
                if (recomputing) continue; // Restart loop to process history
            }

            if (!command_handled) {
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
        }

        if (user_idx < static_cast<int>(prompt_tokens.size())) {
            token = prompt_tokens[user_idx++];
        } else {
            token = next;
        }

        // Add to history (Only the input token is safe to add here)
        if (!recomputing) {
             history_tokens.push_back(token);
        }

        // Check if we are done recomputing
        // Legacy recomputing logic removed as forward_batch handles it instantly.

        if (user_idx >= static_cast<int>(prompt_tokens.size()) && (token == 128009 || token == 128001)) {
            user_turn = true;
        }

        const int flags = (user_idx < static_cast<int>(prompt_tokens.size())) ? FLAG_NO_REPLY : FLAG_NEED_REPLY;
        float *logits = transformer->forward(token, pos, flags);

        // --- AUTOMATIC REWIND CHECK ---
        if (transformer->needs_rewind) {
            transformer->needs_rewind = false;
            std::cout << "\n[System] Automatic load balance: Re-evaluating context..." << std::endl;
            if (!history_tokens.empty()) {
                history_tokens.pop_back(); 
                transformer->forward_batch(history_tokens, 0);
            }
            
            prompt_tokens = history_tokens;
            pos = history_tokens.size();
            user_idx = prompt_tokens.size();
            
            recomputing = false; // Done instantly
            user_turn = false; // Continue generating
            continue;
        }

        if (flags == FLAG_NO_REPLY) {
            next = (user_idx < static_cast<int>(prompt_tokens.size())) ? prompt_tokens[user_idx] : 0;
        } else {
            next = sampler->sample(logits);
        }

        pos++;

        if (user_idx >= static_cast<int>(prompt_tokens.size()) && (next == 128009 || next == 128001)) {
            std::cout << std::endl;
            if (!recomputing) history_tokens.push_back(next); 
        }
        
        if (user_idx >= static_cast<int>(prompt_tokens.size()) && next != 128009 && next != 128001 && next != 128006) {
            if (!recomputing) {
                const std::string &piece = tokenizer->decode(token, next);
                Utils::safe_print(piece);
                std::cout.flush();
                history_tokens.push_back(next); 
            }
        }
    }
    std::cout << std::endl;
}

int main(int argc, char *argv[]) {
    CLI::App app{"LEAP - Optimized Distributed LLM Inference (Ring Topology)"};

    std::string model_path;
    std::string tokenizer_path = "tokenizer.bin";
    app.add_option("model", model_path, "Path to model (.bin)")->required()->check(CLI::ExistingFile);
    app.add_option("-t,--tokenizer", tokenizer_path, "Path to tokenizer (.bin)")
            ->check(CLI::ExistingFile)
            ->capture_default_str();

    std::string role = "single";
    app.add_option("--role", role, "Node role: single, master, worker")
            ->check(CLI::IsMember({"single", "master", "worker"}))
            ->capture_default_str();

    std::string transport_type = "tcp";
    app.add_option("--transport", transport_type, "Transport: tcp, udp, kernel")
            ->check(CLI::IsMember({"tcp", "udp", "kernel"}))
            ->capture_default_str();

    int port = 9999;
    app.add_option("--port", port, "Local port to bind")->capture_default_str();

    std::string host = "0.0.0.0";
    app.add_option("--host", host, "Local host to bind")->capture_default_str();

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

    std::string system_prompt;
    app.add_option("-s,--system", system_prompt, "System prompt");

    bool chat_mode = false;
    app.add_flag("-c,--chat", chat_mode, "Chat mode");

    int n_predict = 4096;
    app.add_option("-n,--n-predict", n_predict, "Max tokens")->capture_default_str();

    float temperature = 1.0f;
    app.add_option("--temp", temperature, "Sampling temperature")->capture_default_str();

    float top_p = 0.9f;
    app.add_option("--top-p", top_p, "Top-p (nucleus) sampling threshold (0.0-1.0)")->capture_default_str();

    unsigned int seed = 0;
    app.add_option("--seed", seed, "Random seed for reproducibility (0 = use current time)")->capture_default_str();

    CLI11_PARSE(app, argc, argv);

    try {
        if (seed == 0) seed = static_cast<unsigned int>(std::time(nullptr));
        auto transformer = Transformer::create(model_path);
        if (end == 0) end = transformer->config.n_layers;

        auto dist_role = DistributedMode::Single;
        if (role == "master") dist_role = DistributedMode::Master;
        else if (role == "worker") dist_role = DistributedMode::Worker;

        if (dist_role == DistributedMode::Master) {
            if (split <= 0 || split >= transformer->config.n_layers) {
                throw std::runtime_error("Master split layer must be in (0, n_layers)");
            }
        } else if (dist_role == DistributedMode::Worker) {
            if (split >= end || end > transformer->config.n_layers) {
                throw std::runtime_error("Worker config error: 0 <= split < end <= n_layers");
            }
        }

        std::unique_ptr<Transport> transport = nullptr;
        if (dist_role != DistributedMode::Single) {
            if (next_host.empty()) throw std::runtime_error("--next-host is required for distributed mode");

            if (transport_type == "tcp") {
                transport = std::make_unique<TcpTransport>(host, port, next_host, next_port);
            } else if (transport_type == "udp") {
                transport = std::make_unique<UdpTransport>(host, port, next_host, next_port);
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
        // If this node processes the final layers, it is the tail of the pipeline
        dist_config.is_tail = (end == transformer->config.n_layers);

        transformer->set_distributed_config(dist_config);

        if (dist_role == DistributedMode::Worker) {
            transformer->worker_loop();
            return 0;
        }

        // Clamp n_predict to model's maximum sequence length
        if (n_predict > transformer->config.seq_len) {
            n_predict = transformer->config.seq_len;
        }

        Tokenizer tokenizer(tokenizer_path, transformer->config.vocab_size);
        Sampler sampler(transformer->config.vocab_size, temperature, top_p, seed);

        if (chat_mode) chat(transformer.get(), &tokenizer, &sampler, prompt, system_prompt, n_predict);
        else generate(transformer.get(), &tokenizer, &sampler, prompt, n_predict);
    } catch (const std::exception &e) {
        std::cerr << "[Error] " << e.what() << std::endl;
        return 1;
    }
    return 0;
}