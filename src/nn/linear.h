
#pragma once

#include <stdbool.h>

#include "../tensor.h"

struct PicoContext;

struct PicoLinear {
    int in_features;
    int out_features;
    struct PicoTensor *weights; // Shape: [in_features, out_features]
    struct PicoTensor *bias;    // Shape: [out_features, 1]
};

struct PicoSwiGLUFFN {
    int model_dim;
    int hidden_dim;
    float dropout_p;
    struct PicoLinear *gate_proj;
    struct PicoLinear *up_proj;
    struct PicoLinear *down_proj;
};

struct PicoLinear *pico_nn_linear_init(struct PicoContext *ctx, char *name, int in_features, int out_features, bool bias);
struct PicoTensor *pico_nn_linear_forward(struct PicoContext *ctx, struct PicoLinear *layer, struct PicoTensor *input);
void pico_nn_linear_free(struct PicoLinear *linear);

struct PicoTensor *pico_nn_fused_swiglu(struct PicoContext *ctx, struct PicoTensor *gate, struct PicoTensor *up, float dropout_p);

struct PicoSwiGLUFFN *
pico_nn_swiglu_ffn_init(struct PicoContext *ctx, char *name, int model_dim, int hidden_dim, float dropout_p, bool bias);
struct PicoTensor *pico_nn_swiglu_ffn_forward(struct PicoContext *ctx, struct PicoSwiGLUFFN *ffn, struct PicoTensor *input);
void pico_nn_swiglu_ffn_free(struct PicoSwiGLUFFN *ffn);
