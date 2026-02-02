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
        if (user_idx < prompt_tokens.size()) {
            flags = FLAG_NO_REPLY;
        }

        float *logits = transformer->forward(token, pos, flags);
        
        if (flags == FLAG_NO_REPLY) {
            if (user_idx < prompt_tokens.size()) {
                next = prompt_tokens[user_idx];
            } else {
                next = 0;
            }
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
    CLI::App app{"LEAP Inference Engine - High-performance distributed LLM inference"};
    app.description("A specialized runner for LEAP models supporting single-node and distributed pipelined inference.\n"
                    "Example (Single Node):\n"
                    "  ./inference model.bin -p \"Hello, how are you?\"\n\n"
                    "Example (Distributed Master):\n"
                    "  ./inference model.bin --role master --next-host 192.168.1.10 --next-port 9999 --split 16\n\n"
                    "Example (Distributed Worker):\n"
                    "  ./inference model.bin --role worker --port 9999 --next-host 192.168.1.11 --next-port 9999 --split 16 --end 32");
    
    argv = app.ensure_utf8(argv);

    // Core arguments
    std::string model_path;
    std::string tokenizer_path = "tokenizer.bin";
    app.add_option("model", model_path, "Path to the model checkpoint file (.bin)")
        ->required()
        ->check(CLI::ExistingFile);

    // General Configuration
    auto *gen_group = app.add_option_group("General", "Basic inference settings");
    std::string mode = "generate";
    bool chat_mode = false;
    gen_group->add_flag("-c,--chat", chat_mode, "Run in interactive chat mode");
    gen_group->add_option("-t,--tokenizer", tokenizer_path, "Path to the tokenizer file")
        ->check(CLI::ExistingFile)
        ->capture_default_str();
    int n_predict = 4096;
    gen_group->add_option("-n,--n-predict", n_predict, "Maximum number of tokens to generate (0 = model max)")
        ->capture_default_str();
    unsigned long long rng_seed = 0;
    gen_group->add_option("--seed", rng_seed, "Random seed for reproducibility (0 = use current time)")
        ->capture_default_str();

    // Sampling Configuration
    auto *sample_group = app.add_option_group("Sampling", "Tokens selection parameters");
    float temperature = 1.0f;
    sample_group->add_option("--temp", temperature, "Temperature for sampling (higher = more creative, 0.0 = greedy)")
        ->check(CLI::NonNegativeNumber)
        ->capture_default_str();
    float topp = 0.9f;
    sample_group->add_option("--top-p", topp, "Top-P (nucleus) sampling threshold")
        ->check(CLI::Range(0.0, 1.0))
        ->capture_default_str();

    // Input
    auto *input_group = app.add_option_group("Input", "Prompt configuration");
    std::string prompt;
    input_group->add_option("-p,--prompt", prompt, "Initial input prompt for the model");
    std::string system_prompt;
    input_group->add_option("--system", system_prompt, "System prompt for chat mode");

    // Distributed Inference
    auto *dist_group = app.add_option_group("Distributed", "Configuration for multi-node pipeline parallelism");
    std::string role = "single";
    dist_group->add_option("--role", role, "Node role in the cluster")
        ->check(CLI::IsMember({"single", "master", "worker"}))
        ->capture_default_str();
    std::string transport_type = "tcp";
    dist_group->add_option("--transport", transport_type, "Network transport protocol")
        ->check(CLI::IsMember({"tcp", "udp", "kernel"}))
        ->capture_default_str();
    
    std::string host = "0.0.0.0";
    dist_group->add_option("--host", host, "Local IP to bind for incoming connections (Workers)")
        ->capture_default_str();
    int port = 9999;
    dist_group->add_option("--port", port, "Local port to bind for incoming connections (Workers)")
        ->check(CLI::Range(1, 65535))
        ->capture_default_str();
    
    std::string next_host = "";
    dist_group->add_option("--next-host", next_host, "IP address of the next node in the pipeline");
    int next_port = 0;
    dist_group->add_option("--next-port", next_port, "Port of the next node in the pipeline")
        ->check(CLI::Range(0, 65535));
    
    int split_layer = 0;
    dist_group->add_option("--split", split_layer, "Layer index to start processing on this node")
        ->capture_default_str();
    int end_layer = 0;
    dist_group->add_option("--end", end_layer, "Layer index to stop processing (exclusive, 0 = until end)")
        ->capture_default_str();

    std::string master_host = "";
    dist_group->add_option("--master-host", master_host, "Master IP address (required for kernel transport)");

    CLI11_PARSE(app, argc, argv);

    // Post-processing
    if (chat_mode) mode = "chat";
    if (rng_seed == 0) rng_seed = static_cast<unsigned int>(std::time(nullptr));
    if (n_predict < 0) n_predict = 0;

    try {
        auto transformer = Transformer::create(model_path);

        std::cout << "Model loaded: " << model_path << std::endl;
        std::cout << "Architecture: "
                << transformer->config.n_layers << " layers, "
                << transformer->config.dim << " dim, "
                << transformer->config.n_heads << " heads, "
                << transformer->config.vocab_size << " vocab, "
                << transformer->config.seq_len << " context" << std::endl;

        if (n_predict == 0 || n_predict > transformer->config.seq_len) {
            n_predict = transformer->config.seq_len;
        }
        
        if (end_layer == 0) end_layer = transformer->config.n_layers;

        // Setup Distributed Mode
        DistributedMode dist_role = DistributedMode::Single;
        if (role == "master") dist_role = DistributedMode::Master;
        else if (role == "worker") dist_role = DistributedMode::Worker;

        std::unique_ptr<Transport> transport = nullptr;

        if (dist_role == DistributedMode::Master) {
            if (split_layer <= 0 || split_layer >= transformer->config.n_layers) {
                throw std::runtime_error("Invalid --split layer for master. Must be > 0 and < n_layers.");
            }
            // Warning: No validation of worker config (gap check)
            std::cerr << "Warning: Ensure the first worker is configured to start at layer " << split_layer 
                      << " to avoid gaps or overlaps." << std::endl;

            if (next_host.empty() || next_port == 0) {
                throw std::runtime_error("Master role requires --next-host and --next-port to connect to workers.");
            }
            
            if (transport_type == "tcp") {
                transport = std::make_unique<TcpTransport>(next_host, next_port, false, "", 0);
            } else if (transport_type == "udp") {
                transport = std::make_unique<UdpTransport>(host, port, false, next_host, next_port);
            } else {
                throw std::runtime_error("Master role does not support 'kernel' transport (Workers only).");
            }
        } else if (dist_role == DistributedMode::Worker) {
            if (split_layer <= 0 || split_layer >= transformer->config.n_layers) {
                throw std::runtime_error("Invalid --split layer for worker.");
            }
            if (end_layer <= split_layer) {
                throw std::runtime_error("Invalid range: --end layer must be greater than --split layer for worker.");
            }
            
            if (transport_type == "tcp") {
                transport = std::make_unique<TcpTransport>(host, port, true, next_host, next_port);
            } else if (transport_type == "udp") {
                transport = std::make_unique<UdpTransport>(host, port, true, next_host, next_port);
            } else if (transport_type == "kernel") {
#ifndef __linux__
                throw std::runtime_error("Kernel transport is only supported on Linux.");
#else
                std::string target = master_host.empty() ? host : master_host;
                transport = std::make_unique<KernelTransport>(target, port, next_host, next_port);
#endif
            }
        }

        if (transport) {
            transport->initialize();
        }

        DistributedConfig dist_config;
        dist_config.mode = dist_role;
        dist_config.split_layer = split_layer;
        dist_config.end_layer = end_layer;
        dist_config.transport = transport.get();
        dist_config.next_ip = next_host;
        dist_config.next_port = next_port;
        dist_config.is_tail = (dist_role == DistributedMode::Worker && next_host.empty());
        
        transformer->set_distributed_config(dist_config);

        if (dist_role == DistributedMode::Worker) {
            std::cout << "Node started as Worker [Layers " << split_layer << " to " << end_layer << "]" << std::endl;
            transformer->worker_loop();
            return 0;
        }

        // Master or Single mode proceeds
        Tokenizer tokenizer(tokenizer_path, transformer->config.vocab_size);
        Sampler sampler(transformer->config.vocab_size, temperature, topp, rng_seed);

        if (mode == "generate") {
            generate(transformer.get(), &tokenizer, &sampler, prompt, n_predict);
        } else if (mode == "chat") {
            chat(transformer.get(), &tokenizer, &sampler, prompt, system_prompt, n_predict);
        }
    } catch (const std::exception &e) {
        std::cerr << "\n[Error] " << e.what() << std::endl;
        return 1;
    }

    return 0;
}