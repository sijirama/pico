#include <math.h>
#include <stdint.h>
#include <stdio.h>

#include "../arena.h"
#include "../ctx.h"
#include "../devices/backend.h"
#include "../kernels/cuda/cuda_ops.h"
#include "../tensor.h"
#include "autograd.h"
#include "loss.h"

static int pico_cross_entropy_shapes_are_valid(struct PicoTensor *logits, struct PicoTensor *targets) {
    if(logits == NULL || targets == NULL || logits->ndim < 1) {
        return 0;
    }

    int64_t classes = logits->shape[logits->ndim - 1];
    if(classes <= 0 || logits->numel % classes != 0) {
        return 0;
    }

    int64_t rows = logits->numel / classes;
    if(targets->numel != rows) {
        return 0;
    }

    if(logits->ndim == 1) {
        return targets->numel == 1;
    }

    if(targets->ndim != logits->ndim - 1) {
        return 0;
    }

    for(int i = 0; i < targets->ndim; i++) {
        if(targets->shape[i] != logits->shape[i]) {
            return 0;
        }
    }

    return 1;
}

static int64_t pico_cross_entropy_logits_base_for_row(struct PicoTensor *logits, int64_t row) {
    if(logits->ndim == 1) {
        return 0;
    }

    int64_t base = 0;
    int64_t rem = row;
    for(int d = logits->ndim - 2; d >= 0; d--) {
        int64_t coord = rem % logits->shape[d];
        rem /= logits->shape[d];
        base += coord * logits->strides[d];
    }
    return base;
}

static int pico_cross_entropy_targets_are_valid(struct PicoTensor *logits, struct PicoTensor *targets) {
    int64_t classes = logits->shape[logits->ndim - 1];
    for(int64_t i = 0; i < targets->numel; i++) {
        int64_t target = (int64_t)targets->data[i];
        if(target < 0 || target >= classes || targets->data[i] != (float)target) {
            return 0;
        }
    }
    return 1;
}

static void pico_cross_entropy_forward(struct PicoTensor *out, struct PicoTensor *logits, struct PicoTensor *targets,
                                       enum PicoCrossEntropyReductionType reduction) {
    int64_t classes = logits->shape[logits->ndim - 1];
    int64_t rows = logits->numel / classes;
    float loss = 0.0f;

    for(int64_t row = 0; row < rows; row++) {
        int64_t target = (int64_t)targets->data[row];
        int64_t base = pico_cross_entropy_logits_base_for_row(logits, row);

        float max_logit = logits->data[base];
        for(int64_t c = 1; c < classes; c++) {
            float value = logits->data[base + c * logits->strides[logits->ndim - 1]];
            if(value > max_logit) {
                max_logit = value;
            }
        }

        float sum_exp = 0.0f;
        for(int64_t c = 0; c < classes; c++) {
            float value = logits->data[base + c * logits->strides[logits->ndim - 1]];
            sum_exp += expf(value - max_logit);
        }

        float target_logit = logits->data[base + target * logits->strides[logits->ndim - 1]];
        loss += (max_logit + logf(sum_exp)) - target_logit;
    }

    if(reduction == PICO_CE_MEAN) {
        loss /= rows;
    }

    out->data[0] = loss;
}

struct PicoCrossEntropyLoss *pico_cross_entropy_loss_init(
    struct PicoContext *ctx,
    enum PicoCrossEntropyReductionType reduction) {
    struct Arena *arena = pico_context_arena(ctx);
    if(arena == NULL) {
        fprintf(stderr, "PicoArenaError: no arena available for cross entropy loss allocation\n");
        return NULL;
    }

    struct PicoCrossEntropyLoss *ce = (struct PicoCrossEntropyLoss *)arena_alloc(arena, sizeof(struct PicoCrossEntropyLoss));
    if(ce == NULL) {
        return NULL;
    }
    ce->reduction = reduction;
    return ce;
}

struct PicoTensor *pico_cross_entropy_loss(
    struct PicoContext *ctx,
    struct PicoCrossEntropyLoss *ce,
    struct PicoTensor *logits,
    struct PicoTensor *targets) {
    if(ctx == NULL || ce == NULL || logits == NULL || targets == NULL) {
        return NULL;
    }

    if(!pico_require_same_backend(logits, targets, "cross_entropy_loss")) {
        return NULL;
    }

    if(!pico_cross_entropy_shapes_are_valid(logits, targets)) {
        fprintf(stderr, "[Pico] Error: cross entropy expects logits [..., classes] and targets [...]\n");
        return NULL;
    }

    if(logits->backend == PICO_BACKEND_CPU && !pico_cross_entropy_targets_are_valid(logits, targets)) {
        fprintf(stderr, "[Pico] Error: cross entropy targets must be integer class ids in range\n");
        return NULL;
    }

    struct Arena *arena = pico_context_arena(ctx);
    if(arena == NULL) {
        fprintf(stderr, "PicoArenaError: no arena available for cross entropy loss output allocation\n");
        return NULL;
    }

    int64_t scalar_shape[] = {1};
    struct PicoTensor *out = pico_create_tensor_on(ctx, logits->backend, scalar_shape, 1);
    if(out == NULL) {
        return NULL;
    }

    if(logits->backend == PICO_BACKEND_CPU) {
        pico_cross_entropy_forward(out, logits, targets, ce->reduction);
    } else if(logits->backend == PICO_BACKEND_CUDA) {
        if(!pico_cuda_cross_entropy(logits, targets, out, ce->reduction)) {
            return NULL;
        }
    } else {
        fprintf(stderr, "PicoBackendError: cross_entropy_loss unknown backend\n");
        return NULL;
    }
    out->_backward = pico_cross_entropy_loss_backward;
    out->op_param = ce->reduction;

    out->parents = arena_alloc(arena, sizeof(struct PicoTensor *) * 2);
    if(out->parents == NULL) {
        return NULL;
    }
    out->parents[0] = logits;
    out->parents[1] = targets;
    out->num_parents = 2;

    out->ndim = 0;
    out->shape = NULL;
    out->strides = NULL;
    out->numel = 1;

    return out;
}
