#include "Tokenizer.h"
#include <iostream>
#include <vector>
#include <string>

#include <CLI/CLI.hpp>

int main(const int argc, char **argv) {
    CLI::App app{"LEAP Tokenizer Exporter"};

    std::string model_path;
    std::string output_path;

    app.add_option("model_path", model_path, "Path to the tokenizer model file")
        ->required()
        ->check(CLI::ExistingFile);

    app.add_option("-o,--output", output_path, "Optional output file path for the binary export");

    CLI11_PARSE(app, argc, argv);

    std::cout << "Loading model from: " << model_path << std::endl;

    try {
        const Tokenizer::Tokenizer tokenizer(model_path);
        std::cout << "Exporting binary model..." << std::endl;
        if (!output_path.empty()) {
             std::cout << "Output path set to: " << output_path << std::endl;
        }
        tokenizer.export_tokenized_binary_file(output_path);
        std::cout << "Export complete." << std::endl;
    } catch (const std::exception &e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
