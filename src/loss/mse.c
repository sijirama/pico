#include <math.h>
#include <stdint.h>

#include "arena.h"
#include "loss.h"
#include "loss/autograd.h"
#include "tensor.h"

void pico_mse_loss_mean(struct PicoTensor* out, struct PicoTensor* prediction,
                        struct PicoTensor* actuals);

void pico_mse_loss_sum(struct PicoTensor* out, struct PicoTensor* prediction,
                       struct PicoTensor* actuals);

struct PicoMSELoss* pico_mse_loss_init(struct Arena* arena, enum PicoMSEReductionType reduction) {
    arena = arena_resolve(arena);
    if(arena == NULL) {
        fprintf(stderr, "PicoArenaError: no arena available for mse loss allocation\n");
        return NULL;
    }

    struct PicoMSELoss* mse = (struct PicoMSELoss*)arena_alloc(arena, sizeof(struct PicoMSELoss));
    mse->reduction = reduction;
    return mse;
}

struct PicoTensor* pico_mse_loss(struct Arena* arena, struct PicoMSELoss* mse, struct PicoTensor* predictions,
                                 struct PicoTensor* actuals) {
    if(!pico_tensor_shapes_are_equal(predictions, actuals)) {
        fprintf(stderr, "[Pico] Error: MSE predictions and actuals must have the same shape\n");
        return NULL;
    }

    if(predictions->backend != actuals->backend) {
        fprintf(stderr, "[Pico] Error: PicoTensor backends are not compatible!\n");
        return NULL;
    }

    arena = arena_resolve(arena);
    if(arena == NULL) {
        fprintf(stderr, "PicoArenaError: no arena available for mse loss output allocation\n");
        return NULL;
    }
    struct PicoTensor* out = pico_create_tensor(arena, predictions->shape, predictions->ndim);

    switch(mse->reduction) {
        case SUM:
            pico_mse_loss_sum(out, predictions, actuals);
            out->_backward = pico_mse_loss_sum_backward;
            break;
        default:
            pico_mse_loss_mean(out, predictions, actuals);
            out->_backward = pico_mse_loss_mean_backward;
            break;
    }

    out->parents = arena_alloc(arena, sizeof(struct PicoTensor*) * 2);
    out->parents[0] = predictions;
    out->parents[1] = actuals;
    out->num_parents = 2;

    out->ndim = 0;
    out->shape = NULL;
    out->numel = 1;

    return out;
}

void pico_mse_loss_mean(struct PicoTensor* out, struct PicoTensor* prediction,
                        struct PicoTensor* actuals) {
    float loss = 0;
    for(int i = 0; i < prediction->numel; i++) {
        loss += powf((prediction->data[i] - actuals->data[i]), 2.0f);
    }

    loss = loss / prediction->numel;

    out->data[0] = loss;
}

void pico_mse_loss_sum(struct PicoTensor* out, struct PicoTensor* prediction,
                       struct PicoTensor* actuals) {
    float loss = 0;
    for(int i = 0; i < prediction->numel; i++) {
        loss += powf((prediction->data[i] - actuals->data[i]), 2.0f);
    }

    out->data[0] = loss;
}
