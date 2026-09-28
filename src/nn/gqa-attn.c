#include "attn.h"

struct PicoAttn *pico_nn_gqa_attn_init(
    struct PicoContext *ctx,
    char *name,
    int embed_dim,
    int num_heads,
    int d_k,
    int group_size) {
    struct PicoAttn *attn =
        pico_nn_attn_init(ctx, name, embed_dim, num_heads, d_k);
    attn->gqa_group_size = group_size;
    return attn;
}

struct PicoTensor *pico_nn_gqa_attn_forward(
    struct PicoContext *ctx, struct PicoAttn *attn, struct PicoTensor *input) {}
