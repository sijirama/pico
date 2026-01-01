#pragma once

#define MAX_DIMS 8

struct TensorOps {
    void (*matmul)(void* x);
    void (*mean)();
};

struct Tensor {
    struct TensorOps* ops;  // operations for the tensor
    float* data;
    int shape[MAX_DIMS];
    int strides[MAX_DIMS];  // The "jump" needed in memory
    int ndim;
};

void tensor_to_cpu(struct Tensor* t);
void tensor_to_gpu(struct Tensor* t);
