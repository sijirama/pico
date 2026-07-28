
#pragma once
#include "tensor.h"

struct PicoContext;

// ==================== MSE

enum PicoMSEReductionType { MEAN, SUM, NONE };

struct PicoMSELoss {
    enum PicoMSEReductionType reduction;
};

struct PicoMSELoss* pico_mse_loss_init(struct PicoContext* ctx, enum PicoMSEReductionType reduction);
struct PicoTensor* pico_mse_loss(struct PicoContext* ctx, struct PicoMSELoss* mse, struct PicoTensor* predictions,
                                 struct PicoTensor* actuals);
