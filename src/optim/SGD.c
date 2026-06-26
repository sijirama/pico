/**
 *
 *
 *
 *
 *
 *
 * **/

#include <stdlib.h>

#include "lib/pico_vector.h"
#include "optim.h"
#include "tensor.h"

struct PicoOptimSGD* pico_optim_sgd_init(float lr) {
    struct PicoVec params;
    pico_vec_init(&params, 25);
    struct PicoOptimSGD* optim = (struct PicoOptimSGD*)calloc(1, sizeof(struct PicoOptimSGD));
    optim->params = params;
    optim->lr = lr;
    return optim;
}

void pico_optim_sgd_add(struct PicoOptimSGD* optim, struct PicoTensor* param) {
    pico_vec_push(&optim->params, param);
}

void pico_optim_sgd_step(struct PicoOptimSGD* optim) {
    struct PicoTensor* tensor = NULL;
    for(int i = 0; i < optim->params.size; i++) {
        tensor = optim->params.data[i];
        for(int j = 0; j < tensor->numel; j++) {
            tensor->data[j] -= optim->lr * tensor->grad[j];
        }
    }
}

void pico_optim_sgd_zero_grad(struct PicoOptimSGD* optim) {
    struct PicoTensor* tensor = NULL;
    for(int i = 0; i < optim->params.size; i++) {
        tensor = optim->params.data[i];
        for(int j = 0; j < tensor->numel; j++) {
            tensor->grad[j] = 0;
        }
    }
}

void pico_optim_sgd_free(struct PicoOptimSGD* optim) {
    pico_vec_free(&optim->params);
    free(optim);
}
