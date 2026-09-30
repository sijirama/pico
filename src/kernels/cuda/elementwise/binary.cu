#include "../cuda_common.cuh"

#include <stdlib.h>

#define PICO_DEFINE_CUDA_BINARY_KERNEL(name, expr)                                                             \
__global__ static void pico_cuda_##name##_kernel(const float* a, const float* b, float* out,                  \
                                                 const int64_t* a_shape, const int64_t* b_shape,              \
                                                 const int64_t* a_strides, const int64_t* b_strides,          \
                                                 const int64_t* out_strides, int ndim, int64_t n) {           \
    int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;                                               \
    if(i >= n) {                                                                                              \
        return;                                                                                               \
    }                                                                                                         \
    int64_t rem = i;                                                                                          \
    int64_t a_offset = 0;                                                                                     \
    int64_t b_offset = 0;                                                                                     \
    for(int d = 0; d < ndim; d++) {                                                                           \
        int64_t coord = rem / out_strides[d];                                                                 \
        rem %= out_strides[d];                                                                                \
        if(a_shape[d] != 1) {                                                                                 \
            a_offset += coord * a_strides[d];                                                                 \
        }                                                                                                     \
        if(b_shape[d] != 1) {                                                                                 \
            b_offset += coord * b_strides[d];                                                                 \
        }                                                                                                     \
    }                                                                                                         \
    out[i] = (expr);                                                                                          \
}

PICO_DEFINE_CUDA_BINARY_KERNEL(add, a[a_offset] + b[b_offset])
PICO_DEFINE_CUDA_BINARY_KERNEL(sub, a[a_offset] - b[b_offset])
PICO_DEFINE_CUDA_BINARY_KERNEL(mul, a[a_offset] * b[b_offset])
PICO_DEFINE_CUDA_BINARY_KERNEL(div, a[a_offset] / b[b_offset])

typedef void (*PicoCudaBinaryKernel)(const float*, const float*, float*, const int64_t*, const int64_t*,
                                     const int64_t*, const int64_t*, const int64_t*, int, int64_t);

static bool pico_cuda_binary(struct PicoTensor* a, struct PicoTensor* b, struct PicoTensor* out,
                             PicoCudaBinaryKernel kernel, const char* name) {
    if(!pico_cuda_tensor_ready(a, name, "a") || !pico_cuda_tensor_ready(b, name, "b") ||
       !pico_cuda_tensor_ready(out, name, "out")) {
        return false;
    }

    int ndim = out->ndim;
    int64_t* h_a_shape = (int64_t*)malloc(ndim * sizeof(int64_t));
    int64_t* h_b_shape = (int64_t*)malloc(ndim * sizeof(int64_t));
    int64_t* h_a_strides = (int64_t*)malloc(ndim * sizeof(int64_t));
    int64_t* h_b_strides = (int64_t*)malloc(ndim * sizeof(int64_t));
    if(h_a_shape == nullptr || h_b_shape == nullptr || h_a_strides == nullptr || h_b_strides == nullptr) {
        fprintf(stderr, "PicoCudaError: %s host metadata allocation failed\n", name);
        free(h_a_shape);
        free(h_b_shape);
        free(h_a_strides);
        free(h_b_strides);
        return false;
    }

    int a_pad = ndim - a->ndim;
    int b_pad = ndim - b->ndim;
    for(int d = 0; d < ndim; d++) {
        if(d < a_pad) {
            h_a_shape[d] = 1;
            h_a_strides[d] = 0;
        } else {
            h_a_shape[d] = a->shape[d - a_pad];
            h_a_strides[d] = a->strides[d - a_pad];
        }

        if(d < b_pad) {
            h_b_shape[d] = 1;
            h_b_strides[d] = 0;
        } else {
            h_b_shape[d] = b->shape[d - b_pad];
            h_b_strides[d] = b->strides[d - b_pad];
        }
    }

    int64_t *d_a_shape = nullptr, *d_b_shape = nullptr, *d_a_strides = nullptr, *d_b_strides = nullptr;
    int64_t* d_out_strides = nullptr;
    size_t meta_bytes = ndim * sizeof(int64_t);
    bool ok = pico_cuda_ok(cudaMalloc((void**)&d_a_shape, meta_bytes), "binary cudaMalloc a_shape") &&
              pico_cuda_ok(cudaMalloc((void**)&d_b_shape, meta_bytes), "binary cudaMalloc b_shape") &&
              pico_cuda_ok(cudaMalloc((void**)&d_a_strides, meta_bytes), "binary cudaMalloc a_strides") &&
              pico_cuda_ok(cudaMalloc((void**)&d_b_strides, meta_bytes), "binary cudaMalloc b_strides") &&
              pico_cuda_ok(cudaMalloc((void**)&d_out_strides, meta_bytes), "binary cudaMalloc out_strides");
    if(!ok) {
        free(h_a_shape); free(h_b_shape); free(h_a_strides); free(h_b_strides);
        cudaFree(d_a_shape); cudaFree(d_b_shape); cudaFree(d_a_strides); cudaFree(d_b_strides);
        cudaFree(d_out_strides);
        return false;
    }

    ok = pico_cuda_ok(cudaMemcpy(d_a_shape, h_a_shape, meta_bytes, cudaMemcpyHostToDevice), "binary copy a_shape") &&
         pico_cuda_ok(cudaMemcpy(d_b_shape, h_b_shape, meta_bytes, cudaMemcpyHostToDevice), "binary copy b_shape") &&
         pico_cuda_ok(cudaMemcpy(d_a_strides, h_a_strides, meta_bytes, cudaMemcpyHostToDevice), "binary copy a_strides") &&
         pico_cuda_ok(cudaMemcpy(d_b_strides, h_b_strides, meta_bytes, cudaMemcpyHostToDevice), "binary copy b_strides") &&
         pico_cuda_ok(cudaMemcpy(d_out_strides, out->strides, meta_bytes, cudaMemcpyHostToDevice), "binary copy out_strides");
    free(h_a_shape); free(h_b_shape); free(h_a_strides); free(h_b_strides);
    if(!ok) {
        cudaFree(d_a_shape); cudaFree(d_b_shape); cudaFree(d_a_strides); cudaFree(d_b_strides);
        cudaFree(d_out_strides);
        return false;
    }

    int threads = 256;
    int blocks = (int)((out->numel + threads - 1) / threads);
    kernel<<<blocks, threads>>>(a->data, b->data, out->data, d_a_shape, d_b_shape, d_a_strides,
                                d_b_strides, d_out_strides, ndim, out->numel);
    ok = pico_cuda_ok(cudaDeviceSynchronize(), name);
    cudaFree(d_a_shape); cudaFree(d_b_shape); cudaFree(d_a_strides); cudaFree(d_b_strides);
    cudaFree(d_out_strides);
    return ok;
}

extern "C" bool pico_cuda_add(struct PicoTensor* a, struct PicoTensor* b, struct PicoTensor* out) {
    return pico_cuda_binary(a, b, out, pico_cuda_add_kernel, "add");
}

extern "C" bool pico_cuda_sub(struct PicoTensor* a, struct PicoTensor* b, struct PicoTensor* out) {
    return pico_cuda_binary(a, b, out, pico_cuda_sub_kernel, "sub");
}

extern "C" bool pico_cuda_mul(struct PicoTensor* a, struct PicoTensor* b, struct PicoTensor* out) {
    return pico_cuda_binary(a, b, out, pico_cuda_mul_kernel, "mul");
}

extern "C" bool pico_cuda_div(struct PicoTensor* a, struct PicoTensor* b, struct PicoTensor* out) {
    return pico_cuda_binary(a, b, out, pico_cuda_div_kernel, "div");
}
