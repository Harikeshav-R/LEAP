#include <iostream>
#include <string>
#include <stdexcept>

#include <CLI/CLI.hpp>

#include "Export.h"
#include "Loader.h"
#include <model/Transformer.h>

// -----------------------------------------------------------------------------
// Core Logic

// Python: def model_export(model, filepath, version):
void model_export(const Model::Transformer &model, const std::string &filepath, const int version) {
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
    CLI::App app{"LEAP Model Exporter"};
    app.footer("Example:\n  ./export output.bin --meta-llama /path/to/llama-model-folder --version 2");

    std::string filepath;
    std::string meta_llama_path;
    int version = 1;

    // Positional argument: output file
    app.add_option("filepath", filepath, "The output filepath for the binary model")
            ->required();

    // Required flag: input model directory
    app.add_option("--meta-llama", meta_llama_path,
                   "Path to the Meta Llama model directory (containing params.json and .safetensors)")
            ->required()
            ->check(CLI::ExistingDirectory);

    // Optional flag: version
    app.add_option("--version", version, "The export format version (1=fp32, 2=int8)")
            ->default_val(1)
            ->check(CLI::Range(1, 2));

    CLI11_PARSE(app, argc, argv);

    try {
        // 2. Load Model        // Note: C++ LibTorch functions typically throw exceptions on failure
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
