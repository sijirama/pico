#include "autograd.h"

#include <stdlib.h>

void backward_add(float* grad_output, struct Tensor* result, float** grad_for_parents) {
    // For z = x + y: dz/dx = 1, dz/dy = 1
    // So both parents get grad_output

    struct Tensor* x = result->parents[0];
    struct Tensor* y = result->parents[1];

    grad_for_parents[0] = malloc(x->numel * sizeof(float));
    grad_for_parents[1] = malloc(y->numel * sizeof(float));

    for(int i = 0; i < result->numel; i++) {
        grad_for_parents[0][i] = grad_output[i];
        grad_for_parents[1][i] = grad_output[i];
    }
}
void backward_mul(float* grad_output, struct Tensor* result, float** grad_for_parents) {
    // For z = x + y: dz/dx = y, dz/dy = x
    // So both parents get grad_output

    struct Tensor* x = result->parents[0];
    struct Tensor* y = result->parents[1];

    float* x_value = result->parents_values[0];
    float* y_value = result->parents_values[1];

    grad_for_parents[0] = malloc(x->numel * sizeof(float));
    grad_for_parents[1] = malloc(y->numel * sizeof(float));

    for(int i = 0; i < result->numel; i++) {
        grad_for_parents[0][i] = grad_output[i] * y_value[i];
        grad_for_parents[1][i] = grad_output[i] * x_value[i];
    }
}

void backward_sub(float* grad_output, struct Tensor* result, float** grad_for_parents) {
    // For z = x - y: dz/dx = 1, dz/dy = - 1
    // So both parents get grad_output

    struct Tensor* x = result->parents[0];
    struct Tensor* y = result->parents[1];

    grad_for_parents[0] = malloc(x->numel * sizeof(float));
    grad_for_parents[1] = malloc(y->numel * sizeof(float));

    for(int i = 0; i < result->numel; i++) {
        grad_for_parents[0][i] = grad_output[i];
        grad_for_parents[1][i] = -grad_output[i];
    }
}

void backward_div(float* grad_output, struct Tensor* result, float** grad_for_parents) {
    // For z = x - y: dz/dx = 1/y, dz/dy = - 1
    // So both parents get grad_output

    struct Tensor* x = result->parents[0];
    struct Tensor* y = result->parents[1];

    float* x_value = result->parents_values[0];
    float* y_value = result->parents_values[1];

    grad_for_parents[0] = malloc(x->numel * sizeof(float));
    grad_for_parents[1] = malloc(y->numel * sizeof(float));

    for(int i = 0; i < result->numel; i++) {
        grad_for_parents[0][i] = grad_output[i] * 1 / y_value[i];
        grad_for_parents[1][i] = grad_output[i] * -(x_value[i] / (y_value[i] * y_value[i]));
    }
}

void tensor_backward(struct Tensor* t, float* grad_output) {
    if(t->grad_op == NONE) {
        if(t->requires_grad) {
            // accumulate if we want to save the grad
            for(int i = 0; i < t->numel; i++) {
                t->grad[i] += grad_output[i];
            }
        };
        return;  // return if it's a leaf node pls, if it's a leaf node then grad_op will be none
    };

    float** grad_for_parents = malloc(t->num_parents * sizeof(float*));

    if(t->grad_op == ADD)
        backward_add(grad_output, t, grad_for_parents);
    if(t->grad_op == MUL)
        backward_mul(grad_output, t, grad_for_parents);
    if(t->grad_op == SUB)
        backward_sub(grad_output, t, grad_for_parents);
    if(t->grad_op == DIV)
        backward_div(grad_output, t, grad_for_parents);

    for(int parent = 0; parent < t->num_parents; parent++) {
        float* grad = grad_for_parents[parent];
        struct Tensor* parent_pointer = t->parents[parent];
        tensor_backward(parent_pointer, grad);
    }

    free(grad_for_parents);
}
