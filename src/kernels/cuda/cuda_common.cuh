#pragma once

#include "cuda_ops.h"

#include <cuda_runtime.h>
#include <stdio.h>

static inline bool pico_cuda_ok(cudaError_t err, const char* what) {
    if(err == cudaSuccess) {
        return true;
    }

    fprintf(stderr, "PicoCudaError: %s failed: %s\n", what, cudaGetErrorString(err));
    return false;
}

static inline bool pico_cuda_tensor_ready(struct PicoTensor* tensor, const char* op_name, const char* arg_name) {
    if(tensor == nullptr) {
        fprintf(stderr, "PicoCudaError: %s received NULL %s tensor\n", op_name, arg_name);
        return false;
    }

    if(tensor->backend != PICO_BACKEND_CUDA) {
        fprintf(stderr, "PicoCudaError: %s expected %s tensor to already be CUDA\n", op_name, arg_name);
        return false;
    }

    if(tensor->data == nullptr || tensor->grad == nullptr) {
        fprintf(stderr, "PicoCudaError: %s received unallocated CUDA %s tensor\n", op_name, arg_name);
        return false;
    }

    return true;
}
