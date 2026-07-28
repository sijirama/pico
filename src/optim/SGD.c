/**
 *
 *
 *
 *
 *
 *
 * **/

#include <stdlib.h>

#include "ctx.h"
#include "optim.h"
#include "tensor.h"

struct PicoOptimSGD* pico_optim_sgd_init(float lr) {
    struct PicoOptimSGD* optim = (struct PicoOptimSGD*)calloc(1, sizeof(struct PicoOptimSGD));
    optim->lr = lr;
    return optim;
}

void pico_optim_sgd_step(struct PicoContext* ctx, struct PicoOptimSGD* optim) {
    if(ctx == NULL || optim == NULL) {
        return;
    }

    struct PicoTensor* tensor = NULL;
    for(int i = 0; i < ctx->params.size; i++) {
        tensor = ctx->params.data[i];
        for(int j = 0; j < tensor->numel; j++) {
            tensor->data[j] -= optim->lr * tensor->grad[j];
        }
    }
}

void pico_optim_sgd_zero_grad(struct PicoContext* ctx, struct PicoOptimSGD* optim) {
    if(ctx == NULL || optim == NULL) {
        return;
    }

    struct PicoTensor* tensor = NULL;
    for(int i = 0; i < ctx->params.size; i++) {
        tensor = ctx->params.data[i];
        for(int j = 0; j < tensor->numel; j++) {
            tensor->grad[j] = 0;
        }
    }
}

void pico_optim_sgd_free(struct PicoOptimSGD* optim) {
    free(optim);
}
