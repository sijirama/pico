/*
 * for better details onbackward functions check out ../autograd.h
 */

#pragma once
#include <math.h>

#include "../kernels/cuda/cuda_ops.h"
#include "../tensor.h"

static inline void pico_mse_loss_mean_backward(struct PicoTensor *self) {
    struct PicoTensor *prediction = self->parents[0];
    struct PicoTensor *actuals = self->parents[1];

    float upstream = self->grad[0]; // the loss is a single scalar
    int64_t N = prediction->numel;

    for(int64_t i = 0; i < N; i++) {

        // local sensitivity: d(loss)/d(pred_i) = (2/N) * (pred_i - actual_i)
        float local = (2.0f / N) * (prediction->data[i] - actuals->data[i]);

        prediction->grad[i] += local * upstream;
        actuals->grad[i] += -local * upstream;
    }
}

static inline void pico_mse_loss_sum_backward(struct PicoTensor *self) {
    struct PicoTensor *prediction = self->parents[0];
    struct PicoTensor *actuals = self->parents[1];

    float upstream = self->grad[0]; // the loss is a single scalar
    int64_t N = prediction->numel;

    for(int64_t i = 0; i < N; i++) {

        // local sensitivity: d(loss)/d(pred_i) = 2 * (pred_i - actual_i)
        float local = (2.0f) * (prediction->data[i] - actuals->data[i]);

        prediction->grad[i] += local * upstream;
        actuals->grad[i] += -local * upstream;
    }
}

static inline int64_t pico_ce_logits_base_for_row(struct PicoTensor *logits, int64_t row) {
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

static inline void pico_cross_entropy_loss_backward(struct PicoTensor *self) {
    struct PicoTensor *logits = self->parents[0];
    struct PicoTensor *targets = self->parents[1];

    if(self->backend == PICO_BACKEND_CUDA) {
        pico_cuda_cross_entropy_backward(self, logits, targets, (int)self->op_param);
        return;
    }

    int64_t classes = logits->shape[logits->ndim - 1];
    int64_t rows = logits->numel / classes;
    float scale = self->op_param == 1 ? 1.0f : 1.0f / rows;
    float upstream = self->grad[0];

    for(int64_t row = 0; row < rows; row++) {
        int64_t target = (int64_t)targets->data[row];
        if(target < 0 || target >= classes) {
            continue;
        }

        int64_t base = pico_ce_logits_base_for_row(logits, row);
        float max_logit = logits->data[base];
        for(int64_t c = 1; c < classes; c++) {
            float value = logits->data[base + c * logits->strides[logits->ndim - 1]];
            if(value > max_logit) {
                max_logit = value;
            }
        }

        float denom = 0.0f;
        for(int64_t c = 0; c < classes; c++) {
            float value = logits->data[base + c * logits->strides[logits->ndim - 1]];
            denom += expf(value - max_logit);
        }

        for(int64_t c = 0; c < classes; c++) {
            int64_t idx = base + c * logits->strides[logits->ndim - 1];
            float prob = expf(logits->data[idx] - max_logit) / denom;
            float one_hot = c == target ? 1.0f : 0.0f;
            logits->grad[idx] += (prob - one_hot) * scale * upstream;
        }
    }
}
