#pragma once

#include "tensor.h"

struct Tensor * Linear(struct Tensor * x);
struct Tensor * Sigmoid(struct Tensor * x);
struct Tensor * TanH(struct Tensor * x);
struct Tensor * Relu(struct Tensor * x);
struct Tensor * LRelu(struct Tensor * x);
struct Tensor * Softmax(struct Tensor * x);
