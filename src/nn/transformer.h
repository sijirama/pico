#pragma once

#include "../tensor.h"
#include "linear.h"
#include "self-attn.h"

struct PicoContext;

struct PicoTransformer {
    int embed_dim;
    int num_heads;
    int d_k;
    int hidden_dim;
    struct PicoAttn *attn;
    struct PicoLinear *ff1;
    struct PicoLinear *ff2;
};

struct PicoTransformer *pico_nn_transformer_init(
    struct PicoContext *ctx,
    char *name,
    int embed_dim,
    int num_heads,
    int d_k,
    int hidden_dim);

struct PicoTensor *pico_nn_transformer_forward(
    struct PicoContext *ctx,
    struct PicoTransformer *block,
    struct PicoTensor *input);

void pico_nn_transformer_free(struct PicoTransformer *block);
