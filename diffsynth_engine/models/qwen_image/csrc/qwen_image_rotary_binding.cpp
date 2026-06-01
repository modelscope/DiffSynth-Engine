#include <torch/extension.h>

torch::Tensor rotary_emb_forward_cuda(torch::Tensor x, torch::Tensor freqs_cis);
std::vector<torch::Tensor> modulate_forward_cuda(torch::Tensor x, torch::Tensor mod_params);
std::vector<torch::Tensor> modulate_indexed_forward_cuda(
    torch::Tensor x,
    torch::Tensor mod_params,
    torch::Tensor index
);

torch::Tensor rotary_emb_forward(torch::Tensor x, torch::Tensor freqs_cis) {
    TORCH_CHECK(x.is_cuda(), "x must be a CUDA tensor");
    TORCH_CHECK(freqs_cis.is_cuda(), "freqs_cis must be a CUDA tensor");
    TORCH_CHECK(x.is_contiguous(), "x must be contiguous");
    TORCH_CHECK(freqs_cis.is_contiguous(), "freqs_cis must be contiguous");
    TORCH_CHECK(x.dim() == 4, "x must have shape [B, S, H, D]");
    TORCH_CHECK(freqs_cis.dim() == 2, "freqs_cis must have shape [S, D/2]");
    TORCH_CHECK(x.size(1) == freqs_cis.size(0), "x sequence dimension must match freqs_cis");
    TORCH_CHECK((x.size(3) % 2) == 0, "last dim of x must be even");
    TORCH_CHECK(freqs_cis.scalar_type() == at::kComplexFloat, "freqs_cis must be complex64");
    TORCH_CHECK(
        (x.scalar_type() == at::kFloat) || (x.scalar_type() == at::kHalf) || (x.scalar_type() == at::kBFloat16),
        "x dtype must be float32, float16, or bfloat16"
    );
    TORCH_CHECK(x.size(3) / 2 == freqs_cis.size(1), "freqs_cis second dim must be D/2");
    return rotary_emb_forward_cuda(x, freqs_cis);
}

std::vector<torch::Tensor> modulate_forward(torch::Tensor x, torch::Tensor mod_params) {
    TORCH_CHECK(x.is_cuda(), "x must be a CUDA tensor");
    TORCH_CHECK(mod_params.is_cuda(), "mod_params must be a CUDA tensor");
    TORCH_CHECK(x.is_contiguous(), "x must be contiguous");
    TORCH_CHECK(mod_params.is_contiguous(), "mod_params must be contiguous");
    TORCH_CHECK(x.dim() == 3, "x must have shape [B, S, D]");
    TORCH_CHECK(mod_params.dim() == 2, "mod_params must have shape [B, 3D]");
    TORCH_CHECK(x.size(0) == mod_params.size(0), "mod_params batch must equal x batch");
    TORCH_CHECK(mod_params.size(1) == x.size(2) * 3, "mod_params last dim must be 3 * D");
    TORCH_CHECK(mod_params.scalar_type() == x.scalar_type(), "mod_params dtype must match x dtype");
    TORCH_CHECK(
        (x.scalar_type() == at::kFloat) || (x.scalar_type() == at::kHalf) || (x.scalar_type() == at::kBFloat16),
        "x dtype must be float32, float16, or bfloat16"
    );
    return modulate_forward_cuda(x, mod_params);
}

std::vector<torch::Tensor> modulate_indexed_forward(torch::Tensor x, torch::Tensor mod_params, torch::Tensor index) {
    TORCH_CHECK(x.is_cuda(), "x must be a CUDA tensor");
    TORCH_CHECK(mod_params.is_cuda(), "mod_params must be a CUDA tensor");
    TORCH_CHECK(index.is_cuda(), "index must be a CUDA tensor");
    TORCH_CHECK(x.is_contiguous(), "x must be contiguous");
    TORCH_CHECK(mod_params.is_contiguous(), "mod_params must be contiguous");
    TORCH_CHECK(index.is_contiguous(), "index must be contiguous");
    TORCH_CHECK(x.dim() == 3, "x must have shape [B, S, D]");
    TORCH_CHECK(mod_params.dim() == 2, "mod_params must have shape [2B, 3D]");
    TORCH_CHECK(index.dim() == 3, "index must have shape [B or 1, S, 1]");
    TORCH_CHECK(index.size(1) == x.size(1), "index sequence dim must match x sequence dim");
    TORCH_CHECK(index.size(2) == 1, "index last dim must be 1");
    TORCH_CHECK(mod_params.size(0) == x.size(0) * 2, "mod_params batch must be 2 * x batch");
    TORCH_CHECK(mod_params.size(1) == x.size(2) * 3, "mod_params last dim must be 3 * D");
    TORCH_CHECK(mod_params.scalar_type() == x.scalar_type(), "mod_params dtype must match x dtype");
    TORCH_CHECK(index.scalar_type() == at::kLong || index.scalar_type() == at::kInt, "index must be int32 or int64");
    TORCH_CHECK(
        (x.scalar_type() == at::kFloat) || (x.scalar_type() == at::kHalf) || (x.scalar_type() == at::kBFloat16),
        "x dtype must be float32, float16, or bfloat16"
    );
    return modulate_indexed_forward_cuda(x, mod_params, index);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("rotary_emb_forward", &rotary_emb_forward, "Qwen rotary embedding forward (CUDA)");
    m.def("modulate_forward", &modulate_forward, "Qwen modulation forward (CUDA)");
    m.def("modulate_indexed_forward", &modulate_indexed_forward, "Qwen indexed modulation forward (CUDA)");
}
