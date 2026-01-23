#include "Utils.h"

namespace Export {
    void serialize_fp32(std::ofstream &out, const torch::Tensor &tensor) {
        // Equivalent to: d = tensor.detach().cpu().view(-1).to(torch.float32).numpy()
        const torch::Tensor d = tensor.detach().cpu().view({-1}).to(torch::kFloat32);

        // In C++, we access the raw data pointer directly rather than using struct.pack
        const auto data_ptr = d.data_ptr<float>();
        const auto num_bytes = d.numel() * sizeof(float);

        out.write(reinterpret_cast<const char *>(data_ptr), num_bytes);
    }

    void serialize_int8(std::ofstream &out, const torch::Tensor &tensor) {
        // Equivalent to: d = tensor.detach().cpu().view(-1).numpy().astype(np.int8)
        const auto d = tensor.detach().cpu().view({-1}).to(torch::kInt8);

        // Access raw data pointer
        const auto data_ptr = d.data_ptr<int8_t>();
        const auto num_bytes = d.numel() * sizeof(int8_t);

        out.write(reinterpret_cast<const char *>(data_ptr), num_bytes);
    }

    std::tuple<torch::Tensor, torch::Tensor, float> quantize_q80(const torch::Tensor &w_in, int64_t group_size) {
        // assert w.numel() % group_size == 0
        TORCH_CHECK(w_in.numel() % group_size == 0, "Tensor numel must be divisible by group_size");

        // w = w.float()
        torch::Tensor w = w_in.to(torch::kFloat32);

        // w = w.reshape(-1, group_size)
        w = w.reshape({-1, group_size});

        // find the max in each group: wmax = torch.abs(w).max(dim=1).values
        // In LibTorch, max returns a tuple (values, indices)
        const auto w_abs = torch::abs(w);
        const auto max_res = torch::max(w_abs, /*dim=*/1);
        const auto wmax = std::get<0>(max_res);

        // calculate the scaling factor such that float = quant * scale
        // scale = wmax / 127.0
        auto scale = wmax / 127.0;

        // scale into range [-127, 127]
        // quant = w / scale[:, None]
        // LibTorch broadcasting requires unsqueeze to match dimensions
        const auto quant = w / scale.unsqueeze(1);

        // round to nearest integer: int8val = torch.round(quant).to(torch.int8)
        auto int8val = torch::round(quant).to(torch::kInt8);

        // dequantize by rescaling
        // fp32val = (int8val.float() * scale[:, None]).view(-1)
        const auto fp32val = (int8val.to(torch::kFloat32) * scale.unsqueeze(1)).view({-1});

        // fp32valr = fp32val.reshape(-1, group_size)
        const auto fp32valr = fp32val.reshape({-1, group_size});

        // calculate the max error in each group
        // err = torch.abs(fp32valr - w).max(dim=1).values
        const auto diff = torch::abs(fp32valr - w);
        const auto err_res = torch::max(diff, /*dim=*/1);
        const auto err = std::get<0>(err_res);

        // find the max error across all groups
        // maxerr = err.max().item()
        auto maxerr = err.max().item<float>();

        return std::make_tuple(int8val, scale, maxerr);
    }
} // namespace Export
