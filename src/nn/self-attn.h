#pragma once

#include "../ctx.h"
#include "../tensor.h"

struct PicoContext;

struct PicoAttn {
    struct PicoTensor* Q;  // [embed_dim, num_heads * d_k]
    struct PicoTensor* K;  // [embed_dim, num_heads * d_k]
    struct PicoTensor* V;  // [embed_dim, num_heads * d_k]
    struct PicoTensor* O;  // [num_heads * d_k, embed_dim]
    int embed_dim;
    int num_of_heads;
    int d_k;
};

struct PicoAttn* pico_nn_attn_init(struct PicoContext* ctx, char* name, int embed_dim, int num_heads, int d_k);
void pico_nn_attn_apply_rope(struct PicoTensor* tensor, int num_heads, int d_k);
struct PicoTensor* pico_nn_attn_forward(struct PicoContext* ctx, struct PicoAttn* attn, struct PicoTensor* input);
void pico_nn_attn_free(struct PicoAttn* attn);
