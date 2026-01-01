#include "tensor.h"

#include <stdio.h>

// cpu_ops prototypes
void tensor_matmul_cpu(struct Tensor* x, struct Tensor* y);
void tensor_add_cpu(struct Tensor* x, struct Tensor* y);
void tensor_mean_cpu(struct Tensor* x);

// ops tables
static const struct TensorOps tensor_ops_cpu = {tensor_matmul_cpu, tensor_add_cpu, tensor_mean_cpu};
static const struct TensorOps tensor_ops_gpu = {tensor_matmul_cpu, tensor_add_cpu,
                                                tensor_mean_cpu};  // use cpu functions for now

// helper functions
int tensor_ndim(struct Tensor* t) {
    return t->ndim;
}
int* tensor_shape(struct Tensor* t) {
    return t->shape;
}
float tensor_get_nd(struct Tensor* t, int* coords) {
    int final_index = 0;
    for(int i = 0; i < t->ndim; i++) {
        final_index += coords[i] * t->strides[i];
    }
    return t->data[final_index];
}
void tensor_update_strides(struct Tensor* t) {
    int current_stride = 1;
    for(int i = t->ndim - 1; i >= 0; i--) {
        t->strides[i] = current_stride;
        current_stride *= t->shape[i];
    }
}

// move to device
void tensor_to_cpu(struct Tensor* t) {
    t->ops = (struct TensorOps*)&tensor_ops_cpu;
}
void tensor_to_gpu(struct Tensor* t) {
    t->ops = (struct TensorOps*)&tensor_ops_gpu;
}

// CPU OPS
void tensor_matmul_cpu(struct Tensor* x, struct Tensor* y) {
    printf("CPU_MATMUL\n");
};
void tensor_add_cpu(struct Tensor* x, struct Tensor* y) {
    printf("CPU_MATMUL\n");
};
void tensor_mean_cpu(struct Tensor* x) {
    printf("CPU_CONCAT\n");
};
