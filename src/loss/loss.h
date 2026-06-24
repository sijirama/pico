
#pragma once
#include "tensor.h"

enum PicoMSEReductionType { MEAN, SUM, NONE };

struct PicoMSELoss {
    enum PicoMSEReductionType reduction;
};


struct PicoMSELoss* pico_mse_loss_init(enum PicoMSEReductionType reduction);
struct PicoTensor* pico_mse_loss(struct PicoMSELoss * mse, struct PicoTensor* predictions, struct PicoTensor* actuals);
