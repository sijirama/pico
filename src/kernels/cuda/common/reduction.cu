#include "../cuda_common.cuh"

__global__ static void pico_cuda_mean_all_kernel(const float* __restrict__ input, float* __restrict__ out, int64_t n) {
    extern __shared__ float shared[];

    float sum = 0.0f;
    for(int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x; i < n; i += (int64_t)gridDim.x * blockDim.x) {
        sum += input[i];
    }

    shared[threadIdx.x] = sum;
    __syncthreads();

    for(int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if(threadIdx.x < stride) {
            shared[threadIdx.x] += shared[threadIdx.x + stride];
        }
        __syncthreads();
    }

    if(threadIdx.x == 0) {
        out[blockIdx.x] = shared[0] / (float)n;
    }
}

__global__ static void pico_cuda_finish_mean_all_kernel(const float* __restrict__ partials, float* __restrict__ out,
                                                        int64_t partial_count) {
    float sum = 0.0f;
    for(int64_t i = 0; i < partial_count; i++) {
        sum += partials[i];
    }
    out[0] = sum;
}

__global__ static void pico_cuda_mean_dim_kernel(const float* __restrict__ input, float* __restrict__ out,
                                                 const int64_t* __restrict__ shape,
                                                 const int64_t* __restrict__ strides, int ndim, int dim,
                                                 int64_t axis_len, int64_t out_numel) {
    int64_t out_i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if(out_i >= out_numel) {
        return;
    }

    int64_t rem = out_i;
    int64_t input_base = 0;
    for(int d = ndim - 1; d >= 0; d--) {
        if(d == dim) {
            continue;
        }

        int64_t coord = rem % shape[d];
        rem /= shape[d];
        input_base += coord * strides[d];
    }

    float sum = 0.0f;
    for(int64_t i = 0; i < axis_len; i++) {
        sum += input[input_base + i * strides[dim]];
    }
    out[out_i] = sum / (float)axis_len;
}

extern "C" bool pico_cuda_mean(struct PicoTensor* input, struct PicoTensor* out, int dim) {
    if(!pico_cuda_tensor_ready(input, "mean", "input") || !pico_cuda_tensor_ready(out, "mean", "out")) {
        return false;
    }

    int threads = 256;
    if(dim == -1) {
        int blocks = (int)((input->numel + threads - 1) / threads);
        if(blocks > 256) {
            blocks = 256;
        }

        float* partials = nullptr;
        if(!pico_cuda_ok(cudaMalloc((void**)&partials, blocks * sizeof(float)), "mean cudaMalloc partials")) {
            return false;
        }

        pico_cuda_mean_all_kernel<<<blocks, threads, threads * sizeof(float)>>>(input->data, partials, input->numel);
        pico_cuda_finish_mean_all_kernel<<<1, 1>>>(partials, out->data, blocks);
        bool ok = pico_cuda_ok(cudaDeviceSynchronize(), "mean");
        cudaFree(partials);
        return ok;
    }

    int ndim = input->ndim;
    int64_t axis_len = input->shape[dim];
    size_t meta_bytes = ndim * sizeof(int64_t);
    int64_t *d_shape = nullptr, *d_strides = nullptr;

    bool ok = pico_cuda_ok(cudaMalloc((void**)&d_shape, meta_bytes), "mean cudaMalloc shape") &&
              pico_cuda_ok(cudaMalloc((void**)&d_strides, meta_bytes), "mean cudaMalloc strides");
    if(!ok) {
        cudaFree(d_shape);
        cudaFree(d_strides);
        return false;
    }

    ok = pico_cuda_ok(cudaMemcpy(d_shape, input->shape, meta_bytes, cudaMemcpyHostToDevice), "mean copy shape") &&
         pico_cuda_ok(cudaMemcpy(d_strides, input->strides, meta_bytes, cudaMemcpyHostToDevice), "mean copy strides");
    if(!ok) {
        cudaFree(d_shape);
        cudaFree(d_strides);
        return false;
    }

    int blocks = (int)((out->numel + threads - 1) / threads);
    pico_cuda_mean_dim_kernel<<<blocks, threads>>>(input->data, out->data, d_shape, d_strides, ndim, dim, axis_len,
                                                   out->numel);
    ok = pico_cuda_ok(cudaDeviceSynchronize(), "mean");
    cudaFree(d_shape);
    cudaFree(d_strides);
    return ok;
}
