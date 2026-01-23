#include <iostream>
#include <string>
#include <stdexcept>

// Include your project headers
#include "Export.h"
#include "Loader.h"
#include <model/Transformer.h>

// -----------------------------------------------------------------------------
// Helper: Simple CLI Argument Parser Structure
struct ProgramArgs {
    std::string filepath;
    std::string meta_llama_path;
    int version = 1;
};

void print_usage(const char *program_name) {
    std::cerr << "usage: " << program_name << " [-h] [--version VERSION] --meta-llama META_LLAMA filepath\n\n"
            << "positional arguments:\n"
            << "  filepath              the output filepath\n\n"
            << "optional arguments:\n"
            << "  -h, --help            show this help message and exit\n"
            << "  --version VERSION     the version to export with (1=fp32, 2=int8) (default: 1)\n"
            << "  --meta-llama          meta llama model path (required)\n";
}

ProgramArgs parse_arguments(const int argc, char *argv[]) {
    ProgramArgs args;
    bool meta_llama_set = false;
    bool filepath_set = false;

    for (int i = 1; i < argc; ++i) {
        if (std::string arg = argv[i]; arg == "-h" || arg == "--help") {
            print_usage(argv[0]);
            std::exit(0);
        } else if (arg == "--version") {
            if (i + 1 < argc) {
                args.version = std::stoi(argv[++i]);
            } else {
                std::cerr << "Error: --version requires an argument.\n";
                std::exit(1);
            }
        } else if (arg == "--meta-llama") {
            if (i + 1 < argc) {
                args.meta_llama_path = argv[++i];
                meta_llama_set = true;
            } else {
                std::cerr << "Error: --meta-llama requires a path argument.\n";
                std::exit(1);
            }
        } else if (arg[0] == '-') {
            std::cerr << "Error: Unknown argument " << arg << "\n";
            print_usage(argv[0]);
            std::exit(1);
        } else {
            // Assume positional argument is filepath
            if (!filepath_set) {
                args.filepath = arg;
                filepath_set = true;
            } else {
                std::cerr << "Error: Too many positional arguments provided.\n";
                print_usage(argv[0]);
                std::exit(1);
            }
        }
    }

    if (!filepath_set) {
        std::cerr << "Error: the following arguments are required: filepath\n";
        print_usage(argv[0]);
        std::exit(1);
    }

    if (!meta_llama_set) {
        std::cerr << "Error: the following arguments are required: --meta-llama\n";
        print_usage(argv[0]);
        std::exit(1);
    }

    return args;
}

// -----------------------------------------------------------------------------
// Core Logic

// Python: def model_export(model, filepath, version):
void model_export(const Model::Transformer &model, const std::string &filepath, int version) {
    if (version == 1) {
        Export::float32_export(model, filepath);
    } else if (version == 2) {
        Export::int8_export(model, filepath);
    } else {
        throw std::invalid_argument("unknown version " + std::to_string(version));
    }
}

// -----------------------------------------------------------------------------
// Main Entry Point

int main(const int argc, char *argv[]) {
    try {
        // 1. Parse Args
        auto [filepath, meta_llama_path, version] = parse_arguments(argc, argv);

        // 2. Load Model
        // Note: C++ LibTorch functions typically throw exceptions on failure 
        // rather than returning None/nullptr, so we wrap in try/catch.
        std::cout << "Loading model from " << meta_llama_path << "..." << std::endl;

        const Model::Transformer model = Export::load_meta_model(meta_llama_path);

        // 3. Export
        std::cout << "Exporting version " << version << " to " << filepath << "..." << std::endl;
        model_export(model, filepath, version);

        std::cout << "Done." << std::endl;
    } catch (const std::exception &e) {
        std::cerr << "Runtime Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
