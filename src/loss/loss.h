
#pragma once
#include "../tensor.h"

struct PicoContext;

// ==================== MSE

enum PicoMSEReductionType { MEAN, SUM, NONE };

struct PicoMSELoss {
    enum PicoMSEReductionType reduction;
};

struct PicoMSELoss* pico_mse_loss_init(struct PicoContext* ctx, enum PicoMSEReductionType reduction);
struct PicoTensor* pico_mse_loss(struct PicoContext* ctx, struct PicoMSELoss* mse, struct PicoTensor* predictions,
                                 struct PicoTensor* actuals);


// ==================== Cross entropy

enum PicoCrossEntropyReductionType { PICO_CE_MEAN, PICO_CE_SUM };

struct PicoCrossEntropyLoss {
    enum PicoCrossEntropyReductionType reduction;
};

struct PicoCrossEntropyLoss* pico_cross_entropy_loss_init(struct PicoContext* ctx,
                                                          enum PicoCrossEntropyReductionType reduction);
struct PicoTensor* pico_cross_entropy_loss(struct PicoContext* ctx, struct PicoCrossEntropyLoss* ce,
                                           struct PicoTensor* logits, struct PicoTensor* targets);
