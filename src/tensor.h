#pragma once
#define MAX_DIMS 8

struct Tensor {
    struct TensorOps* ops;  // operations for the tensor
    float* data;
    int shape[MAX_DIMS];
    int strides[MAX_DIMS];
    int ndim;
};

struct TensorOps {
    void (*matmul)(struct Tensor* x, struct Tensor* y);
    void (*add)(struct Tensor* x, struct Tensor* y);
    void (*mean)(struct Tensor* x);
};

void tensor_to_cpu(struct Tensor* t);
void tensor_to_gpu(struct Tensor* t);

int tensor_ndim(struct Tensor* t);
int* tensor_shape(struct Tensor* t);
float tensor_get_nd(struct Tensor* t, int* coords);
void tensor_update_strides(struct Tensor* t);
