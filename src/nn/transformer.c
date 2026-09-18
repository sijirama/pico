#include "transformer.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "../act/activations.h"
#include "../arena.h"
#include "../ctx.h"
#include "../ops.h"

static char *
pico_transformer_child_name(struct Arena *arena, char *name, char *suffix) {
    size_t len = strlen(name) + strlen(suffix) + 1;
    char *child_name = arena_alloc(arena, len);
    if(child_name == NULL) {
        return NULL;
    }

    strcpy(child_name, name);
    strcat(child_name, suffix);
    return child_name;
}

struct PicoTransformer *pico_nn_transformer_init(
    struct PicoContext *ctx,
    char *name,
    int embed_dim,
    int num_heads,
    int d_k,
    int hidden_dim) {

    if(ctx == NULL || name == NULL || embed_dim <= 0 || num_heads <= 0 ||
       d_k <= 0 || hidden_dim <= 0) {
        return NULL;
    }

    struct Arena *arena = pico_context_arena(ctx);
    if(arena == NULL) {
        fprintf(
            stderr,
            "PicoArenaError: no arena available for transformer init "
            "allocation\n");
        return NULL;
    }

    struct PicoTransformer *block = malloc(sizeof(struct PicoTransformer));
    if(block == NULL) {
        perror("Failed to allocate PicoTransformer");
        return NULL;
    }

    block->embed_dim = embed_dim;
    block->num_heads = num_heads;
    block->d_k = d_k;
    block->hidden_dim = hidden_dim;

    char *attn_name = pico_transformer_child_name(arena, name, ".attn");
    char *ff1_name = pico_transformer_child_name(arena, name, ".ff1");
    char *ff2_name = pico_transformer_child_name(arena, name, ".ff2");

    if(attn_name == NULL || ff1_name == NULL || ff2_name == NULL) {
        free(block);
        return NULL;
    }

    block->attn = pico_nn_attn_init(ctx, attn_name, embed_dim, num_heads, d_k);
    block->ff1 =
        pico_nn_linear_init(ctx, ff1_name, embed_dim, hidden_dim, true);
    block->ff2 =
        pico_nn_linear_init(ctx, ff2_name, hidden_dim, embed_dim, true);

    if(block->attn == NULL || block->ff1 == NULL || block->ff2 == NULL) {
        pico_nn_transformer_free(block);
        return NULL;
    }

    return block;
}

struct PicoTensor *pico_nn_transformer_forward(
    struct PicoContext *ctx,
    struct PicoTransformer *block,
    struct PicoTensor *input) {
    if(ctx == NULL || block == NULL || input == NULL) {
        return NULL;
    }

    struct PicoTensor *attn_out = pico_nn_attn_forward(ctx, block->attn, input);
    if(attn_out == NULL) {
        return NULL;
    }

    struct PicoTensor *x = pico_add(ctx, input, attn_out);
    if(x == NULL) {
        return NULL;
    }

    struct PicoTensor *hidden = pico_nn_linear_forward(ctx, block->ff1, x);
    if(hidden == NULL) {
        return NULL;
    }

    hidden = pico_relu(ctx, hidden);
    if(hidden == NULL) {
        return NULL;
    }

    struct PicoTensor *ff_out = pico_nn_linear_forward(ctx, block->ff2, hidden);
    if(ff_out == NULL) {
        return NULL;
    }

    return pico_add(ctx, x, ff_out);
}

void pico_nn_transformer_free(struct PicoTransformer *block) {
    if(block == NULL) {
        return;
    }

    pico_nn_attn_free(block->attn);
    pico_nn_linear_free(block->ff1);
    pico_nn_linear_free(block->ff2);
    free(block);
}
