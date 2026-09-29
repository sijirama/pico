

#pragma once
#include <math.h>

struct PicoContext;
struct PicoTensor;

static inline float sigmoid(float x) {
    return 1.0f / (1.0f + expf(-x));
}

static inline float silu(float x) {
    return x * sigmoid(x);
}

struct PicoTensor *pico_relu(struct PicoContext *ctx, struct PicoTensor *x);
struct PicoTensor *pico_sigmoid(struct PicoContext *ctx, struct PicoTensor *x);
struct PicoTensor *pico_swiglu(struct PicoContext *ctx, struct PicoTensor *x, struct PicoTensor *gate);
