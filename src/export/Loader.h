#ifndef LEAP_LOADER_H
#define LEAP_LOADER_H

#include <string>
#include <torch/torch.h>

#include <model/Transformer.h>

namespace Export {
    Model::Transformer load_meta_model(const std::string &model_path);
} // namespace Model

#endif //LEAP_LOADER_H