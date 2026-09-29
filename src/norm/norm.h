
#pragma once

#include "../tensor.h"

struct PicoContext;

// ==================== LayerNorm

struct PicoLayerNorm {
    char* name;
    int normalized_dim;
    float eps;
};

struct PicoLayerNorm* pico_nn_layernorm_init(struct PicoContext* ctx, char* name, int normalized_dim, float eps);
struct PicoTensor* pico_nn_layernorm_forward(struct PicoContext* ctx, struct PicoLayerNorm* norm,
                                             struct PicoTensor* input);
void pico_nn_layernorm_free(struct PicoLayerNorm* norm);

// ==================== RMSNorm

struct PicoRMSNorm {
    char* name;
    int normalized_dim;
    float eps;
    struct PicoTensor* weight;
};

struct PicoRMSNorm* pico_nn_rmsnorm_init(struct PicoContext* ctx, char* name, int normalized_dim, float eps);
struct PicoTensor* pico_nn_rmsnorm_forward(struct PicoContext* ctx, struct PicoRMSNorm* norm,
                                           struct PicoTensor* input);
void pico_nn_rmsnorm_free(struct PicoRMSNorm* norm);
