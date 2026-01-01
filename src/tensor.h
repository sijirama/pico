#pragma once

struct Tensor;
struct TensorOps;

struct Tensor {
    struct TensorOps* ops;  // operations for the tensor
    float* data;
    int* shape;
    int* strides;
    int numel;
    int ndim;
};

// High-level API (The Wrappers)
void tensor_matmul(struct Tensor* a, struct Tensor* b);
void tensor_add(struct Tensor* a, struct Tensor* b);
float tensor_mean(struct Tensor* a);
float tensor_sum(struct Tensor* a);
float tensor_max(struct Tensor* a);

struct TensorOps {
    void (*matmul)(struct Tensor* x, struct Tensor* y);
    void (*add)(struct Tensor* x, struct Tensor* y);
    float (*mean)(struct Tensor* x);
    float (*sum)(struct Tensor* x);
    float (*max)(struct Tensor* x);
};

void tensor_to_cpu(struct Tensor* t);
void tensor_to_gpu(struct Tensor* t);


struct Tensor* tensor_create(int* shape, int ndim);
struct Tensor* tensor_from_data(float* existing_data, int* shape, int ndim);
void tensor_free(struct Tensor* t);

int tensor_ndim(struct Tensor* t);
int* tensor_shape(struct Tensor* t);
float tensor_get_nd(struct Tensor* t, int* coords);
void tensor_update_strides(struct Tensor* t);
