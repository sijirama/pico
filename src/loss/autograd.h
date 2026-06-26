/*
 * for better details onbackward functions check out ../autograd.h
 */

#pragma once
#include "tensor.h"


static inline void pico_mse_loss_mean_backward(struct PicoTensor* self) {
    struct PicoTensor* prediction = self->parents[0];
    struct PicoTensor* actuals = self->parents[1];

    float upstream = self->grad[0];  // the loss is a single scalar
    int64_t N = prediction->numel;

    for(int64_t i = 0; i < N; i++) {
        
        // local sensitivity: d(loss)/d(pred_i) = (2/N) * (pred_i - actual_i)
        float local = (2.0f / N) * (prediction->data[i] - actuals->data[i]);

        prediction->grad[i] += local * upstream;
        actuals->grad[i] += -local * upstream;
    }
}

static inline void pico_mse_loss_sum_backward(struct PicoTensor* self) {
    struct PicoTensor* prediction = self->parents[0];
    struct PicoTensor* actuals = self->parents[1];

    float upstream = self->grad[0];  // the loss is a single scalar
    int64_t N = prediction->numel;

    for(int64_t i = 0; i < N; i++) {
        
        // local sensitivity: d(loss)/d(pred_i) = 2 * (pred_i - actual_i)
        float local = (2.0f) * (prediction->data[i] - actuals->data[i]);

        prediction->grad[i] += local * upstream;
        actuals->grad[i] += -local * upstream;
    }
}
