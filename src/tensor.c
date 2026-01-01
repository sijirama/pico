#include "tensor.h"

#include <stdio.h>
#include <stdlib.h>

/* =========================================================================
   BACKEND DISPATCH (Internal)
   ========================================================================= */
void tensor_matmul_cpu(struct Tensor* x, struct Tensor* y);
void tensor_add_cpu(struct Tensor* x, struct Tensor* y);
float tensor_mean_cpu(struct Tensor* x);
float tensor_sum_cpu(struct Tensor* x);
float tensor_max_cpu(struct Tensor* x);

// ops tables
static const struct TensorOps tensor_ops_cpu = {tensor_matmul_cpu, tensor_add_cpu, tensor_mean_cpu,
                                                tensor_sum_cpu, tensor_max_cpu};  // cpu functions
static const struct TensorOps tensor_ops_gpu = {tensor_matmul_cpu, tensor_add_cpu,
                                                tensor_mean_cpu};  // use cpu functions for now

/* =========================================================================
   Create tensors
   ========================================================================= */

struct Tensor* tensor_create(int* shape, int ndim) {
    struct Tensor* t;

    t = (struct Tensor*)malloc(sizeof(struct Tensor));
    if(t == NULL) {
        perror("Memory allocation failed");
        return NULL;
    }

    t->ndim = ndim;

    t->shape = malloc(ndim * sizeof(int));
    t->strides = malloc(ndim * sizeof(int));

    for(int i = 0; i < ndim; i++) {
        t->shape[i] = shape[i];
    }

    int total_elements = 1;
    for(int i = 0; i < ndim; i++) {
        total_elements *= t->shape[i];
    }
    t->data = malloc(total_elements * sizeof(float));

    for(int i = 0; i < total_elements; i++) {
        t->data[i] = 0.0f;
    }

    t->numel = total_elements;

    tensor_update_strides(t);
    tensor_to_cpu(t);
    return t;
}

struct Tensor* tensor_from_data(float* existing_data, int* shape, int ndim) {
    struct Tensor* t = tensor_create(shape, ndim);
    if(t == NULL)
        return NULL;

    for(int i = 0; i < t->numel; i++) {
        t->data[i] = existing_data[i];
    }

    return t;
}

void tensor_free(struct Tensor* t) {
    free(t->data);
    free(t->shape);
    free(t->strides);
    free(t);
}

/* =========================================================================
   PUBLIC API (The Wrappers)
   ========================================================================= */
void tensor_add(struct Tensor* a, struct Tensor* b) {
    if(*a->shape != *b->shape) {
        fprintf(stderr, "Error: Tensor shapes do not match for addition (%d vs %d)\n", *a->shape,
                *b->shape);
        return;
    }
    a->ops->add(a, b);
}
void tensor_matmul(struct Tensor* a, struct Tensor* b) {
    a->ops->matmul(a, b);
}
float tensor_mean(struct Tensor* a) {
    return a->ops->mean(a);
}
float tensor_sum(struct Tensor* a) {
    return a->ops->sum(a);
}
float tensor_max(struct Tensor* a) {
    return a->ops->max(a);
}

/* =========================================================================
   CPU BACKEND IMPLEMENTATION
   ========================================================================= */
void tensor_matmul_cpu(struct Tensor* x, struct Tensor* y) {
    printf("CPU_MATMUL\n");
};

void tensor_add_cpu(struct Tensor* x, struct Tensor* y) {
    printf("CPU_ADD\n");
};

float tensor_max_cpu(struct Tensor* t) {
    float max = t->data[0];
    for(int i = 1; i < t->numel; i++) {
        if(t->data[i] > max) {
            max = t->data[i];
        }
    }
    return max;
};

float tensor_sum_cpu(struct Tensor* t) {
    float sum = 0;
    for(int i = 0; i < t->numel; i++) {
        sum += t->data[i];
    }
    return sum;
};

float tensor_mean_cpu(struct Tensor* t) {
    return tensor_sum(t) / t->numel;
};

/* =========================================================================
   CORE UTILITIES & HELPERS
   ========================================================================= */
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

/* =========================================================================
   DEVICE MANAGEMENT
   ========================================================================= */
void tensor_to_cpu(struct Tensor* t) {
    t->ops = (struct TensorOps*)&tensor_ops_cpu;
}
void tensor_to_gpu(struct Tensor* t) {
    t->ops = (struct TensorOps*)&tensor_ops_gpu;
}
