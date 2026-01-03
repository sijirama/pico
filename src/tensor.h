#pragma once

struct Tensor;
struct TensorOps;

typedef enum {
    FLOAT32,
    INT32,
    FLOAT64
} DataType;

struct Tensor {
    struct TensorOps* ops;  // operations for the tensor
    float* data;
    // Datatype dtype;
    int* shape;
    int* strides;
    int numel;
    int ndim;
};

// High-level API (The Wrappers)
void tensor_matmul(struct Tensor* a, struct Tensor* b);
void tensor_dot(struct Tensor* a, struct Tensor* b);
void tensor_truediv(struct Tensor* a, struct Tensor* b);
void tensor_sub(struct Tensor* a, struct Tensor* b);
void tensor_add(struct Tensor* a, struct Tensor* b);
float tensor_max(struct Tensor* a);
float tensor_sum(struct Tensor* a);
float tensor_mean(struct Tensor* a);

void tensor_reshape(struct Tensor* a, int * shape, int ndim);
void tensor_transpose_2d(struct Tensor* a);
void print_tensor(const char* label, struct Tensor* t);

struct TensorOps {
    void (*matmul)(struct Tensor* x, struct Tensor* y);
    void (*dot)(struct Tensor* x, struct Tensor* y);
    void (*truediv)(struct Tensor* x, struct Tensor* y);
    void (*sub)(struct Tensor* x, struct Tensor* y);
    void (*add)(struct Tensor* x, struct Tensor* y);
    float (*max)(struct Tensor* x);
    float (*sum)(struct Tensor* x);
    float (*mean)(struct Tensor* x);
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
