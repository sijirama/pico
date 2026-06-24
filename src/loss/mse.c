#include <math.h>
#include <stdint.h>

#include "loss.h"
#include "tensor.h"

void pico_mse_loss_mean(struct PicoTensor* out, struct PicoTensor* prediction,
                        struct PicoTensor* actuals);

struct PicoTensor* pico_mse_loss(struct PicoMSELoss* mse, struct PicoTensor* predictions,
                                 struct PicoTensor* actuals) {
    if(predictions->shape != actuals->shape) {
        fprintf(stderr, "[Pico] Error: PicoTensors are not compatible!\n");
        return NULL;
    }

    if(predictions->backend != actuals->backend) {
        fprintf(stderr, "[Pico] Error: PicoTensor backends are not compatible!\n");
        return NULL;
    }

    struct Arena* arena = arena_ctx_current();
    if(arena == NULL) {
        fprintf(stderr, "[Pico] Error: No current arena in context!\n");
        return NULL;
    }
    struct PicoTensor* out = pico_create_tensor(arena, predictions->shape, predictions->ndim);

    switch(mse->reduction) {
        default:
            pico_mse_loss_mean(out, predictions, actuals);
    }

    return out;
}

void pico_mse_loss_mean(struct PicoTensor* out, struct PicoTensor* prediction,
                        struct PicoTensor* actuals) {
    float loss = 0;
    for(int i = 0; i < prediction->numel; i++) {
        loss += powf((prediction->data[i] - actuals->data[i]), 2.0f);
    }

    loss = loss / prediction->numel;
    int64_t shape[] = {};

    out->data[0] = loss;
    out->ndim = 1;
    out->shape = shape;
}
