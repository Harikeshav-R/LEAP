#include "Tokenizer.h"
#include <iostream>
#include <vector>
#include <string>

int main(const int argc, char **argv) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <model_path>" << std::endl;
        return 1;
    }

    const std::string model_path = argv[1];
    std::cout << "Loading model from: " << model_path << std::endl;

    try {
        const Tokenizer tokenizer(model_path);
        std::cout << "Exporting binary model..." << std::endl;
        tokenizer.export_tokenized_binary_file();
        std::cout << "Export complete." << std::endl;
    } catch (const std::exception &e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
