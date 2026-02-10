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
#include <sstream>
#include <thread>
#include <chrono>

using namespace Inference;

void read_stdin(const char *guide, std::string &buffer) {
    std::cout << guide;
    std::getline(std::cin, buffer);
}

// Wait for ACK message from workers after sending resize command.
// The ACK travels around the ring back to the master.
// Returns true if ACK received within timeout, false otherwise.
bool wait_for_resize_ack(Transport* transport, int timeout_ms = 5000) {
    size_t packet_size = transport->get_packet_size();
    if (packet_size == 0) {
        // Fallback: no packet size set, use control header size
        packet_size = sizeof(ControlPacketHeader);
    }
    
    std::vector<char> buffer(packet_size);
    
    auto start = std::chrono::steady_clock::now();
    while (true) {
        // Check for timeout
        auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::steady_clock::now() - start).count();
        if (elapsed >= timeout_ms) {
            std::cerr << "Warning: Resize ACK timeout after " << timeout_ms << "ms" << std::endl;
            return false;
        }
        
        // Receive a packet (blocking)
        transport->recv_prev(buffer.data(), packet_size);
        
        // Check if it's an ACK
        uint16_t magic = 0;
        std::memcpy(&magic, buffer.data(), sizeof(magic));
        if (magic == CONTROL_MAGIC) {
            ControlPacketHeader pkt{};
            std::memcpy(&pkt, buffer.data(), sizeof(pkt));
            if (pkt.msg.type == ControlMessageType::ACK) {
                return true;  // Successfully received ACK
            }
        }
        // Not an ACK - this shouldn't happen in normal operation
        // but continue waiting in case of spurious packets
    }
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

                // Handle slash commands
                if (!user_prompt_str.empty() && user_prompt_str[0] == '/') {
                    std::istringstream iss(user_prompt_str);
                    std::string cmd;
                    iss >> cmd;

                    if (cmd == "/layers") {
                        // Show current layer configuration
                        std::cout << "Current layer config:\n";
                        if (transformer->dist_config.mode == DistributedMode::Master) {
                            std::cout << "  Role: Master\n";
                            std::cout << "  Master layers: 0 to " << transformer->dist_config.split_layer - 1 << "\n";
                            std::cout << "  Worker start layer: " << transformer->dist_config.split_layer << "\n";
                        } else if (transformer->dist_config.mode == DistributedMode::Worker) {
                            std::cout << "  Role: Worker\n";
                            std::cout << "  Processing layers: " << transformer->dist_config.split_layer 
                                      << " to " << transformer->dist_config.end_layer - 1 << "\n";
                        } else {
                            std::cout << "  Role: Single (no distribution)\n";
                            std::cout << "  Processing all " << transformer->config.n_layers << " layers\n";
                        }
                        continue;
                    } else if (cmd == "/resize") {
                        // Check if running in single mode (no workers)
                        if (transformer->dist_config.mode == DistributedMode::Single) {
                            std::cerr << "Error: /resize is only available in distributed mode (master/worker)\n";
                            std::cerr << "  Start with --role master to enable layer resizing\n";
                            continue;
                        }
                        
                        if (!transformer->dist_config.transport) {
                            std::cerr << "Error: No transport available. Cannot resize without worker connection.\n";
                            continue;
                        }
                        
                        // Parse all layer boundaries
                        std::vector<int> boundaries;
                        int boundary;
                        while (iss >> boundary) {
                            boundaries.push_back(boundary);
                        }
                        
                        if (boundaries.empty()) {
                            std::cerr << "Usage: /resize <layer_boundaries...>\n";
                            std::cerr << "  Single worker:  /resize 8 16    (master:0-7, worker:8-15)\n";
                            std::cerr << "  Multi-worker:   /resize 8 12 16 (master:0-7, w0:8-11, w1:12-15)\n";
                            continue;
                        }
                        
                        int n_layers = transformer->config.n_layers;
                        int num_boundaries = static_cast<int>(boundaries.size());
                        
                        // Validate boundaries are increasing and within range
                        bool valid = true;
                        int prev = 0;
                        for (int i = 0; i < num_boundaries && valid; i++) {
                            if (boundaries[i] <= prev || boundaries[i] > n_layers) {
                                std::cerr << "Error: Boundaries must be increasing and <= " << n_layers << "\n";
                                valid = false;
                            }
                            prev = boundaries[i];
                        }
                        if (!valid) continue;
                        
                        // Update master's split_layer (save old value for KV transfer)
                        int old_split = transformer->dist_config.split_layer;
                        transformer->dist_config.split_layer = boundaries[0];
                        
                        if (num_boundaries == 1) {
                            // Single boundary: just update master, worker keeps same end_layer
                            ControlMessage msg{};
                            msg.type = ControlMessageType::RESIZE_LAYERS;
                            msg.split_layer = boundaries[0];
                            msg.end_layer = n_layers;  // Worker goes to end
                            msg.is_tail = true;
                            transformer->dist_config.transport->send_control(msg);
                            
                            // Wait for ACK from worker to ensure resize is applied
                            if (wait_for_resize_ack(transformer->dist_config.transport)) {
                                std::cout << "Resize complete (ACK received).\n";
                            }
                            std::cout << "  Master: layers 0 to " << boundaries[0] - 1 << "\n";
                            std::cout << "  Worker: layers " << boundaries[0] << " to " << n_layers - 1 << " (tail)\n";
                        } else if (num_boundaries == 2) {
                            // Two boundaries: single worker with explicit end
                            ControlMessage msg{};
                            msg.type = ControlMessageType::RESIZE_LAYERS;
                            msg.split_layer = boundaries[0];
                            msg.end_layer = boundaries[1];
                            msg.is_tail = (boundaries[1] == n_layers);
                            transformer->dist_config.transport->send_control(msg);
                            
                            // Wait for ACK from worker
                            if (wait_for_resize_ack(transformer->dist_config.transport)) {
                                std::cout << "Resize complete (ACK received).\n";
                            }
                            std::cout << "  Master: layers 0 to " << boundaries[0] - 1 << "\n";
                            std::cout << "  Worker: layers " << boundaries[0] << " to " << boundaries[1] - 1;
                            if (msg.is_tail) std::cout << " (tail)";
                            std::cout << "\n";
                        } else {
                            // Multiple boundaries: multi-worker chain resize
                            int num_workers = num_boundaries - 1;
                            
                            if (num_workers > MAX_WORKERS) {
                                std::cerr << "Error: Maximum " << MAX_WORKERS << " workers supported\n";
                                continue;
                            }
                            
                            ControlMessage msg{};
                            msg.type = ControlMessageType::RESIZE_CHAIN;
                            msg.worker_index = 0;
                            msg.total_workers = num_workers;
                            
                            int start = boundaries[0];
                            for (int i = 0; i < num_workers; i++) {
                                msg.ranges[i].start_layer = start;
                                msg.ranges[i].end_layer = boundaries[i + 1];
                                msg.ranges[i].is_tail = (boundaries[i + 1] == n_layers);
                                start = boundaries[i + 1];
                            }
                            
                            // Drain any stale packets from the UDP socket before control exchange
                            auto *udp = dynamic_cast<UdpTransport *>(transformer->dist_config.transport);
                            if (udp) udp->drain_stale_packets();
                            
                            std::cerr << "[Resize] Sending RESIZE_CHAIN to " << num_workers << " workers..." << std::endl;
                            transformer->dist_config.transport->send_control(msg);
                            
                            std::cerr << "[Resize] Waiting for ACK..." << std::endl;
                            // Wait for ACK from tail worker (propagates through entire chain)
                            if (wait_for_resize_ack(transformer->dist_config.transport)) {
                                std::cout << "Chain resize complete (ACK received).\n";
                            }
                            std::cout << "  Master: layers 0 to " << boundaries[0] - 1 << "\n";
                            start = boundaries[0];
                            for (int i = 0; i < num_workers; i++) {
                                std::cout << "  Worker " << i << ": layers " << start 
                                          << " to " << boundaries[i + 1] - 1;
                                if (boundaries[i + 1] == n_layers) std::cout << " (tail)";
                                std::cout << "\n";
                                start = boundaries[i + 1];
                            }
                        }
                        
                        // KV Cache Transfer: seamlessly move cache data between nodes
                        if (pos > 0) {
                            const int kv_dim = transformer->config.dim * transformer->config.n_kv_heads / transformer->config.n_heads;
                            std::cout << "Transferring KV cache (" << pos << " positions)..." << std::endl;
                            transformer->initiate_kv_transfer(
                                transformer->get_key_cache(), transformer->get_value_cache(),
                                pos, kv_dim, transformer->config.seq_len,
                                old_split, boundaries[0]);
                            std::cout << "[KV cache transferred - context preserved]\n";
                        }
                        
                        continue;
                    } else if (cmd == "/help") {
                        std::cout << "Available commands:\n";
                        std::cout << "  /layers           - Show current layer distribution\n";
                        std::cout << "  /resize <...>     - Resize layer distribution\n";
                        std::cout << "                      Single worker: /resize 8 16\n";
                        std::cout << "                      Multi-worker:  /resize 8 12 16\n";
                        std::cout << "  /help             - Show this help\n";
                        std::cout << "  exit              - Exit chat\n";
                        continue;
                    } else {
                        std::cerr << "Unknown command: " << cmd << ". Type /help for available commands.\n";
                        continue;
                    }
                }
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

        if (user_idx < static_cast<int>(prompt_tokens.size())) {
            token = prompt_tokens[user_idx++];
        } else {
            token = next;
        }

        if (user_idx >= static_cast<int>(prompt_tokens.size()) && (token == 128009 || token == 128001)) {
            user_turn = true;
        }

        const int flags = (user_idx < static_cast<int>(prompt_tokens.size())) ? FLAG_NO_REPLY : FLAG_NEED_REPLY;
        float *logits = transformer->forward(token, pos, flags);

        if (flags == FLAG_NO_REPLY) {
            next = (user_idx < static_cast<int>(prompt_tokens.size())) ? prompt_tokens[user_idx] : 0;
        } else {
            next = sampler->sample(logits);
        }

        pos++;

        if (user_idx >= static_cast<int>(prompt_tokens.size()) && (next == 128009 || next == 128001)) {
            std::cout << std::endl;
        }
        if (user_idx >= static_cast<int>(prompt_tokens.size()) && next != 128009 && next != 128001 && next != 128006) {
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
        
        // Set packet size on transport for properly padded control messages
        if (transport) {
            size_t packet_size = sizeof(PacketHeader) + transformer->config.dim * sizeof(float);
            transport->set_packet_size(packet_size);
        }

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