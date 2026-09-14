#include "self-attn.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "arena.h"

static char* pico_attn_param_name(struct Arena* arena, char* name, char* suffix) {
    size_t len = strlen(name) + strlen(suffix) + 1;
    char* param_name = arena_alloc(arena, len);
    if(param_name == NULL) {
        return NULL;
    }

    strcpy(param_name, name);
    strcat(param_name, suffix);
    return param_name;
}

struct PicoAttn* pico_nn_attn_init(struct PicoContext* ctx, char* name, int embed_dim, int num_heads, int d_k) {
    if(ctx == NULL || name == NULL || embed_dim <= 0 || num_heads <= 0 || d_k <= 0) {
        return NULL;
    }

    struct Arena* arena = pico_context_arena(ctx);
    if(arena == NULL) {
        fprintf(stderr, "PicoArenaError: no arena available for attention init allocation\n");
        return NULL;
    }

    struct PicoAttn* attn = malloc(sizeof(struct PicoAttn));
    if(attn == NULL) {
        perror("Failed to allocate PicoAttn");
        return NULL;
    }

    attn->embed_dim = embed_dim;
    attn->num_of_heads = num_heads;
    attn->d_k = d_k;

    int head_dim = num_heads * d_k;
    int64_t q_shape[2] = {embed_dim, head_dim};
    int64_t k_shape[2] = {embed_dim, head_dim};
    int64_t v_shape[2] = {embed_dim, head_dim};
    int64_t o_shape[2] = {head_dim, embed_dim};

    char* q_name = pico_attn_param_name(arena, name, ".q_proj.weight");
    char* k_name = pico_attn_param_name(arena, name, ".k_proj.weight");
    char* v_name = pico_attn_param_name(arena, name, ".v_proj.weight");
    char* o_name = pico_attn_param_name(arena, name, ".out_proj.weight");

    if(q_name == NULL || k_name == NULL || v_name == NULL || o_name == NULL) {
        free(attn);
        return NULL;
    }

    attn->Q = pico_param_named(ctx, q_name, q_shape, 2);
    attn->K = pico_param_named(ctx, k_name, k_shape, 2);
    attn->V = pico_param_named(ctx, v_name, v_shape, 2);
    attn->O = pico_param_named(ctx, o_name, o_shape, 2);

    if(attn->Q == NULL || attn->K == NULL || attn->V == NULL || attn->O == NULL) {
        free(attn);
        return NULL;
    }

    return attn;
}

void pico_nn_attn_free(struct PicoAttn* attn) {
    if(attn == NULL) {
        return;
    }

    free(attn);
}
