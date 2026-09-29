#include <math.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>

#include "../ctx.h"
#include "../tensor.h"
#include "optim.h"

#define PICO_ADAM_BETA1 0.9f
#define PICO_ADAM_BETA2 0.999f
#define PICO_ADAM_EPS 1e-8f

static void pico_optim_adam_clear_state(struct PicoOptimAdam *optim) {
    if(optim == NULL) {
        return;
    }

    if(optim->m != NULL) {
        for(int i = 0; i < optim->param_count; i++) {
            free(optim->m[i]);
        }
    }

    if(optim->v != NULL) {
        for(int i = 0; i < optim->param_count; i++) {
            free(optim->v[i]);
        }
    }

    free(optim->params);
    free(optim->m);
    free(optim->v);

    optim->params = NULL;
    optim->m = NULL;
    optim->v = NULL;
    optim->param_count = 0;
    optim->step = 0;
}

static bool pico_optim_adam_state_matches(struct PicoContext *ctx, struct PicoOptimAdam *optim) {
    if(ctx == NULL || optim == NULL || optim->param_count != ctx->params.size) {
        return false;
    }

    for(int i = 0; i < ctx->params.size; i++) {
        if(optim->params[i] != (struct PicoTensor *)ctx->params.data[i]) {
            return false;
        }
    }
    return true;
}

static bool pico_optim_adam_ensure_state(struct PicoContext *ctx, struct PicoOptimAdam *optim) {
    if(ctx == NULL || optim == NULL) {
        return false;
    }

    if(pico_optim_adam_state_matches(ctx, optim)) {
        return true;
    }

    pico_optim_adam_clear_state(optim);

    optim->param_count = ctx->params.size;
    if(optim->param_count == 0) {
        return true;
    }

    optim->params = (struct PicoTensor **)calloc(optim->param_count, sizeof(struct PicoTensor *));
    optim->m = (float **)calloc(optim->param_count, sizeof(float *));
    optim->v = (float **)calloc(optim->param_count, sizeof(float *));
    if(optim->params == NULL || optim->m == NULL || optim->v == NULL) {
        pico_optim_adam_clear_state(optim);
        return false;
    }

    for(int i = 0; i < optim->param_count; i++) {
        struct PicoTensor *tensor = (struct PicoTensor *)ctx->params.data[i];
        optim->params[i] = tensor;
        optim->m[i] = (float *)calloc(tensor->numel, sizeof(float));
        optim->v[i] = (float *)calloc(tensor->numel, sizeof(float));
        if(optim->m[i] == NULL || optim->v[i] == NULL) {
            pico_optim_adam_clear_state(optim);
            return false;
        }
    }

    return true;
}

struct PicoOptimAdam *pico_optim_adam_init(float lr) {
    struct PicoOptimAdam *optim = (struct PicoOptimAdam *)calloc(1, sizeof(struct PicoOptimAdam));
    if(optim == NULL) {
        return NULL;
    }

    optim->lr = lr;
    optim->beta1 = PICO_ADAM_BETA1;
    optim->beta2 = PICO_ADAM_BETA2;
    optim->eps = PICO_ADAM_EPS;
    return optim;
}

void pico_optim_adam_step(struct PicoContext *ctx, struct PicoOptimAdam *optim) {
    if(ctx == NULL || optim == NULL || !pico_optim_adam_ensure_state(ctx, optim)) {
        return;
    }

    optim->step += 1;
    float beta1_correction = 1.0f - powf(optim->beta1, (float)optim->step);
    float beta2_correction = 1.0f - powf(optim->beta2, (float)optim->step);

    for(int i = 0; i < optim->param_count; i++) {
        struct PicoTensor *tensor = optim->params[i];
        if(tensor == NULL || tensor->data == NULL || tensor->grad == NULL) {
            fprintf(stderr, "PicoOptimAdam: param data/grad is not on cpu\n");
            continue;
        }

        for(int j = 0; j < tensor->numel; j++) {
            float grad = tensor->grad[j];
            optim->m[i][j] = optim->beta1 * optim->m[i][j] + (1.0f - optim->beta1) * grad;
            optim->v[i][j] = optim->beta2 * optim->v[i][j] + (1.0f - optim->beta2) * grad * grad;

            float m_hat = optim->m[i][j] / beta1_correction;
            float v_hat = optim->v[i][j] / beta2_correction;
            tensor->data[j] -= optim->lr * m_hat / (sqrtf(v_hat) + optim->eps);
        }
    }
}

void pico_optim_adam_zero_grad(struct PicoContext *ctx, struct PicoOptimAdam *optim) {
    if(ctx == NULL || optim == NULL) {
        return;
    }

    for(int i = 0; i < ctx->params.size; i++) {
        struct PicoTensor *tensor = (struct PicoTensor *)ctx->params.data[i];
        if(tensor == NULL || tensor->grad == NULL) {
            continue;
        }

        for(int j = 0; j < tensor->numel; j++) {
            tensor->grad[j] = 0.0f;
        }
    }
}

void pico_optim_adam_free(struct PicoOptimAdam *optim) {
    if(optim == NULL) {
        return;
    }

    pico_optim_adam_clear_state(optim);
    free(optim);
}
