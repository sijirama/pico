#include "tensor.h"

#include <stdio.h>

// cpu_ops declarations
void tensor_matmul_cpu(void* x);
void tensor_mean_cpu();

// gpu_ops declarations
void tensor_matmul_gpu(void* x);
void tensor_mean_gpu();

// ops tables
static const struct TensorOps tensor_ops_cpu = {tensor_matmul_cpu, tensor_mean_cpu};
static const struct TensorOps tensor_ops_gpu = {tensor_matmul_gpu, tensor_mean_gpu};

void tensor_to_cpu(struct Tensor* t) {
    t->ops = (struct TensorOps*)&tensor_ops_cpu;
}

void tensor_to_gpu(struct Tensor* t) {
    t->ops = (struct TensorOps*)&tensor_ops_gpu;
}

// CPU OPS
void tensor_matmul_cpu(void* x) {
    printf("CPU_MATMUL\n");
};
void tensor_mean_cpu() {
    printf("CPU_CONCAT\n");
};

// GPU OPS
void tensor_matmul_gpu(void* x) {
    printf("GPU_MATMUL\n");
}
void tensor_mean_gpu() {
    printf("GPU_MEAN\n");
}
