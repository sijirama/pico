#pragma once
#include "tensor.h"


struct Tensor;

void backward_add(float* grad_output, struct Tensor* result, float** grad_for_parents);
void backward_sub(float* grad_output, struct Tensor* result, float** grad_for_parents);
void backward_mul(float* grad_output, struct Tensor* result, float** grad_for_parents);
void backward_div(float* grad_output, struct Tensor* result, float** grad_for_parents);
