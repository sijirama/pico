#include "activations.h"

#include <math.h>
#define MAX_RELU(a, b) (((a) > (b)) ? (a) : (b))

struct Tensor* Linear(struct Tensor* x) {
    return tensor_clone(x);
}
struct Tensor* Relu(struct Tensor* x) {
    struct Tensor* t = tensor_clone(x);
    for(int i = 0; i < t->numel; i++) {
        t->data[i] = MAX_RELU(t->data[i], 0);
    }
    return t;
}

struct Tensor* TanH(struct Tensor* x) {
    struct Tensor* t = tensor_clone(x);
    for(int i = 0; i < t->numel; i++) {
        t->data[i] = (exp(t->data[i]) - exp(-t->data[i])) / (exp(t->data[i]) + exp(-t->data[i]));
    }
    return t;
};

struct Tensor* Sigmoid(struct Tensor* x) {
    struct Tensor* t = tensor_clone(x);
    for(int i = 0; i < t->numel; i++) {
        t->data[i] = 1 / (1 + exp(-t->data[i]));
    }
    return t;
}

struct Tensor* Softmax(struct Tensor* x) {
    float sum_exp = 0;
    for(int i = 0; i < x->numel; i++) {
        sum_exp += exp(x->data[i]);
    }
    struct Tensor* t = tensor_clone(x);
    for(int i = 0; i < x->numel; i++) {
        t->data[i] = exp(x->data[i]) / sum_exp;
    }
    return t;
};
