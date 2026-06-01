#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAStream.h>
#include <torch/extension.h>

#include <cuda.h>
#include <cuda_runtime.h>

template <typename scalar_t>
__global__ void rotary_emb_kernel(
    const scalar_t* __restrict__ x,
    const c10::complex<float>* __restrict__ freqs,
    scalar_t* __restrict__ out,
    int64_t bsz,
    int64_t seq,
    int64_t heads,
    int64_t dim
) {
    const int64_t half_dim = dim >> 1;
    const int64_t total_pairs = bsz * seq * heads * half_dim;
    const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= total_pairs) return;

    int64_t t = idx;
    const int64_t pair_idx = t % half_dim;
    t /= half_dim;
    const int64_t head_idx = t % heads;
    t /= heads;
    const int64_t seq_idx = t % seq;
    const int64_t batch_idx = t / seq;

    const int64_t base = ((batch_idx * seq + seq_idx) * heads + head_idx) * dim + (pair_idx << 1);
    const float x0 = static_cast<float>(x[base]);
    const float x1 = static_cast<float>(x[base + 1]);
    const c10::complex<float> f = freqs[seq_idx * half_dim + pair_idx];
    const float fr = f.real();
    const float fi = f.imag();

    const float y0 = x0 * fr - x1 * fi;
    const float y1 = x0 * fi + x1 * fr;
    out[base] = static_cast<scalar_t>(y0);
    out[base + 1] = static_cast<scalar_t>(y1);
}

torch::Tensor rotary_emb_forward_cuda(torch::Tensor x, torch::Tensor freqs_cis) {
    auto out = torch::empty_like(x);
    const int64_t bsz = x.size(0);
    const int64_t seq = x.size(1);
    const int64_t heads = x.size(2);
    const int64_t dim = x.size(3);
    const int64_t total_pairs = bsz * seq * heads * (dim >> 1);

    if (total_pairs == 0) {
        return out;
    }

    constexpr int threads = 256;
    const int blocks = static_cast<int>((total_pairs + threads - 1) / threads);
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::kHalf,
        at::kBFloat16,
        x.scalar_type(),
        "qwen_image_rotary_emb_cuda",
        [&] {
            rotary_emb_kernel<scalar_t><<<blocks, threads, 0, stream>>>(
                x.data_ptr<scalar_t>(),
                freqs_cis.data_ptr<c10::complex<float>>(),
                out.data_ptr<scalar_t>(),
                bsz,
                seq,
                heads,
                dim
            );
        }
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}


template <typename scalar_t>
__global__ void modulate_kernel(
    const scalar_t* __restrict__ x,
    const scalar_t* __restrict__ shift,
    const scalar_t* __restrict__ scale,
    scalar_t* __restrict__ modulated,
    int64_t bsz,
    int64_t seq,
    int64_t dim
) {
    const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const int64_t total = bsz * seq * dim;
    if (idx >= total) return;

    const int64_t d = idx % dim;
    const int64_t t = idx / dim;
    const int64_t b = t / seq;
    const int64_t base_vec = b * dim + d;

    const scalar_t xv = x[idx];
    const scalar_t shiftv = shift[base_vec];
    const scalar_t scalev = scale[base_vec];
    modulated[idx] = xv * (static_cast<scalar_t>(1.0f) + scalev) + shiftv;
}


template <typename scalar_t, typename index_t>
__global__ void modulate_indexed_kernel(
    const scalar_t* __restrict__ x,
    const scalar_t* __restrict__ shift,
    const scalar_t* __restrict__ scale,
    const scalar_t* __restrict__ gate,
    const index_t* __restrict__ index,
    scalar_t* __restrict__ modulated,
    scalar_t* __restrict__ gate_out,
    int64_t bsz,
    int64_t seq,
    int64_t dim,
    int64_t index_bsz
) {
    const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const int64_t total = bsz * seq * dim;
    if (idx >= total) return;

    const int64_t d = idx % dim;
    const int64_t t = idx / dim;
    const int64_t s = t % seq;
    const int64_t b = t / seq;

    const int64_t ib = (index_bsz == 1) ? 0 : b;
    const int64_t index_offset = (ib * seq + s);
    const int64_t select = static_cast<int64_t>(index[index_offset]) == 0 ? b : (b + bsz);
    const int64_t base_vec = select * dim + d;

    const scalar_t xv = x[idx];
    const scalar_t shiftv = shift[base_vec];
    const scalar_t scalev = scale[base_vec];
    modulated[idx] = xv * (static_cast<scalar_t>(1.0f) + scalev) + shiftv;
    gate_out[idx] = gate[base_vec];
}


std::vector<torch::Tensor> modulate_forward_cuda(torch::Tensor x, torch::Tensor mod_params) {
    const int64_t bsz = x.size(0);
    const int64_t seq = x.size(1);
    const int64_t dim = x.size(2);
    auto modulated = torch::empty_like(x);

    const int64_t total = bsz * seq * dim;
    if (total == 0) {
        auto gate_out = mod_params.narrow(1, dim * 2, dim).unsqueeze(1).contiguous();
        return {modulated, gate_out};
    }

    auto shift = mod_params.narrow(1, 0, dim).contiguous();
    auto scale = mod_params.narrow(1, dim, dim).contiguous();
    constexpr int threads = 256;
    const int blocks = static_cast<int>((total + threads - 1) / threads);
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::kHalf,
        at::kBFloat16,
        x.scalar_type(),
        "qwen_image_modulate_cuda",
        [&] {
            modulate_kernel<scalar_t><<<blocks, threads, 0, stream>>>(
                x.data_ptr<scalar_t>(),
                shift.data_ptr<scalar_t>(),
                scale.data_ptr<scalar_t>(),
                modulated.data_ptr<scalar_t>(),
                bsz,
                seq,
                dim
            );
        }
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    auto gate_out = mod_params.narrow(1, dim * 2, dim).unsqueeze(1).contiguous();
    return {modulated, gate_out};
}


std::vector<torch::Tensor> modulate_indexed_forward_cuda(
    torch::Tensor x,
    torch::Tensor mod_params,
    torch::Tensor index
) {
    const int64_t bsz = x.size(0);
    const int64_t seq = x.size(1);
    const int64_t dim = x.size(2);
    const int64_t index_bsz = index.size(0);
    auto modulated = torch::empty_like(x);
    auto gate_out = torch::empty_like(x);

    const int64_t total = bsz * seq * dim;
    if (total == 0) {
        return {modulated, gate_out};
    }

    auto shift = mod_params.narrow(1, 0, dim).contiguous();
    auto scale = mod_params.narrow(1, dim, dim).contiguous();
    auto gate = mod_params.narrow(1, dim * 2, dim).contiguous();
    auto index_2d = index.squeeze(-1);
    constexpr int threads = 256;
    const int blocks = static_cast<int>((total + threads - 1) / threads);
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::kHalf,
        at::kBFloat16,
        x.scalar_type(),
        "qwen_image_modulate_indexed_cuda",
        [&] {
            if (index_2d.scalar_type() == at::kLong) {
                modulate_indexed_kernel<scalar_t, int64_t><<<blocks, threads, 0, stream>>>(
                    x.data_ptr<scalar_t>(),
                    shift.data_ptr<scalar_t>(),
                    scale.data_ptr<scalar_t>(),
                    gate.data_ptr<scalar_t>(),
                    index_2d.data_ptr<int64_t>(),
                    modulated.data_ptr<scalar_t>(),
                    gate_out.data_ptr<scalar_t>(),
                    bsz,
                    seq,
                    dim,
                    index_bsz
                );
            } else {
                modulate_indexed_kernel<scalar_t, int><<<blocks, threads, 0, stream>>>(
                    x.data_ptr<scalar_t>(),
                    shift.data_ptr<scalar_t>(),
                    scale.data_ptr<scalar_t>(),
                    gate.data_ptr<scalar_t>(),
                    index_2d.data_ptr<int>(),
                    modulated.data_ptr<scalar_t>(),
                    gate_out.data_ptr<scalar_t>(),
                    bsz,
                    seq,
                    dim,
                    index_bsz
                );
            }
        }
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {modulated, gate_out};
}
