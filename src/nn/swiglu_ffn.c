#include "linear.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "../act/activations.h"
#include "../arena.h"
#include "../ctx.h"
#include "../tensor_ops.h"

static char *pico_swiglu_ffn_child_name(struct Arena *arena, char *name, const char *suffix) {
    size_t len = strlen(name) + strlen(suffix) + 1;
    char *child_name = arena_alloc(arena, len);
    if(child_name == NULL) {
        return NULL;
    }

    strcpy(child_name, name);
    strcat(child_name, suffix);
    return child_name;
}

struct PicoSwiGLUFFN *
pico_nn_swiglu_ffn_init(struct PicoContext *ctx, char *name, int model_dim, int hidden_dim, float dropout_p, bool bias) {
    if(ctx == NULL || name == NULL || model_dim <= 0 || hidden_dim <= 0) {
        fprintf(stderr, "PicoFFNError: invalid swiglu ffn init args\n");
        return NULL;
    }

    if(dropout_p < 0.0f || dropout_p >= 1.0f) {
        fprintf(stderr, "PicoFFNError: dropout_p must be in [0, 1)\n");
        return NULL;
    }

    struct Arena *arena = pico_context_arena(ctx);
    if(arena == NULL) {
        fprintf(stderr, "PicoArenaError: no arena available for swiglu ffn init allocation\n");
        return NULL;
    }

    struct PicoSwiGLUFFN *ffn = malloc(sizeof(struct PicoSwiGLUFFN));
    if(ffn == NULL) {
        return NULL;
    }

    ffn->model_dim = model_dim;
    ffn->hidden_dim = hidden_dim;
    ffn->dropout_p = dropout_p;
    ffn->gate_proj = pico_nn_linear_init(ctx, pico_swiglu_ffn_child_name(arena, name, ".gate_proj"), model_dim, hidden_dim, bias);
    ffn->up_proj = pico_nn_linear_init(ctx, pico_swiglu_ffn_child_name(arena, name, ".up_proj"), model_dim, hidden_dim, bias);
    ffn->down_proj = pico_nn_linear_init(ctx, pico_swiglu_ffn_child_name(arena, name, ".down_proj"), hidden_dim, model_dim, bias);

    if(ffn->gate_proj == NULL || ffn->up_proj == NULL || ffn->down_proj == NULL) {
        pico_nn_swiglu_ffn_free(ffn);
        return NULL;
    }

    return ffn;
}

struct PicoTensor *pico_nn_swiglu_ffn_forward(struct PicoContext *ctx, struct PicoSwiGLUFFN *ffn, struct PicoTensor *input) {
    if(ctx == NULL || ffn == NULL || input == NULL) {
        return NULL;
    }

    struct PicoTensor *gate = pico_nn_linear_forward(ctx, ffn->gate_proj, input);
    struct PicoTensor *up = pico_nn_linear_forward(ctx, ffn->up_proj, input);
    if(gate == NULL || up == NULL) {
        return NULL;
    }

    struct PicoTensor *hidden = pico_nn_fused_swiglu(ctx, gate, up, ffn->dropout_p);
    if(hidden == NULL) {
        return NULL;
    }

    return pico_nn_linear_forward(ctx, ffn->down_proj, hidden);
}

void pico_nn_swiglu_ffn_free(struct PicoSwiGLUFFN *ffn) {
    if(ffn == NULL) {
        return;
    }

    pico_nn_linear_free(ffn->gate_proj);
    pico_nn_linear_free(ffn->up_proj);
    pico_nn_linear_free(ffn->down_proj);
    free(ffn);
}
