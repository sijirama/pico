#include "../cuda_common.cuh"

#include <math.h>

__global__ static void pico_cuda_softmax_dim_kernel(const float* input, float* output,
                                                    const int64_t* shape, const int64_t* in_strides,
                                                    const int64_t* out_strides, int ndim, int dim,
                                                    int64_t axis_len, int64_t slice_count) {
    int64_t slice = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if(slice >= slice_count) {
        return;
    }

    int64_t rem = slice;
    int64_t input_base = 0;
    int64_t output_base = 0;
    for(int d = ndim - 1; d >= 0; d--) {
        if(d == dim) {
            continue;
        }

        int64_t coord = rem % shape[d];
        rem /= shape[d];
        input_base += coord * in_strides[d];
        output_base += coord * out_strides[d];
    }

    float max_value = input[input_base];
    for(int64_t i = 1; i < axis_len; i++) {
        float value = input[input_base + i * in_strides[dim]];
        if(value > max_value) {
            max_value = value;
        }
    }

    float sum = 0.0f;
    for(int64_t i = 0; i < axis_len; i++) {
        float value = expf(input[input_base + i * in_strides[dim]] - max_value);
        output[output_base + i * out_strides[dim]] = value;
        sum += value;
    }

    for(int64_t i = 0; i < axis_len; i++) {
        output[output_base + i * out_strides[dim]] /= sum;
    }
}

extern "C" bool pico_cuda_softmax(struct PicoTensor* input, struct PicoTensor* out, uint8_t dim) {
    if(!pico_cuda_tensor_ready(input, "softmax", "input") || !pico_cuda_tensor_ready(out, "softmax", "out")) {
        return false;
    }

    int ndim = input->ndim;
    int64_t axis_len = input->shape[dim];
    int64_t slice_count = input->numel / axis_len;
    size_t meta_bytes = ndim * sizeof(int64_t);

    int64_t *d_shape = nullptr, *d_in_strides = nullptr, *d_out_strides = nullptr;
    bool ok = pico_cuda_ok(cudaMalloc((void**)&d_shape, meta_bytes), "softmax cudaMalloc shape") &&
              pico_cuda_ok(cudaMalloc((void**)&d_in_strides, meta_bytes), "softmax cudaMalloc in_strides") &&
              pico_cuda_ok(cudaMalloc((void**)&d_out_strides, meta_bytes), "softmax cudaMalloc out_strides");
    if(!ok) {
        cudaFree(d_shape); cudaFree(d_in_strides); cudaFree(d_out_strides);
        return false;
    }

    ok = pico_cuda_ok(cudaMemcpy(d_shape, input->shape, meta_bytes, cudaMemcpyHostToDevice), "softmax copy shape") &&
         pico_cuda_ok(cudaMemcpy(d_in_strides, input->strides, meta_bytes, cudaMemcpyHostToDevice), "softmax copy in_strides") &&
         pico_cuda_ok(cudaMemcpy(d_out_strides, out->strides, meta_bytes, cudaMemcpyHostToDevice), "softmax copy out_strides");
    if(!ok) {
        cudaFree(d_shape); cudaFree(d_in_strides); cudaFree(d_out_strides);
        return false;
    }

    int threads = 128;
    int blocks = (int)((slice_count + threads - 1) / threads);
    pico_cuda_softmax_dim_kernel<<<blocks, threads>>>(input->data, out->data, d_shape, d_in_strides, d_out_strides,
                                                      ndim, dim, axis_len, slice_count);
    ok = pico_cuda_ok(cudaDeviceSynchronize(), "softmax");
    cudaFree(d_shape); cudaFree(d_in_strides); cudaFree(d_out_strides);
    return ok;
}
