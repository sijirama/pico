#include "norm.h"

#include <stdlib.h>
#include <string.h>
#include <stdio.h>

#include "../arena.h"
#include "../ctx.h"
#include "../ops.h"
#include "../tensor.h"
#include "../tensor_ops.h"

struct PicoLayerNorm* pico_nn_layernorm_init(struct PicoContext* ctx, char* name, int normalized_dim, float eps) {
    if(ctx == NULL || name == NULL || normalized_dim <= 0) {
        return NULL;
    }

    struct Arena* arena = pico_context_arena(ctx);
    if(arena == NULL) {
        fprintf(stderr, "PicoArenaError: no arena available for layernorm init allocation\n");
        return NULL;
    }

    struct PicoLayerNorm* norm = malloc(sizeof(struct PicoLayerNorm));
    if(norm == NULL) {
        perror("Failed to allocate PicoLayerNorm");
        return NULL;
    }

    norm->name = arena_alloc(arena, strlen(name) + 1);
    if(norm->name == NULL) {
        free(norm);
        return NULL;
    }

    strcpy(norm->name, name);
    norm->normalized_dim = normalized_dim;
    norm->eps = eps > 0.0f ? eps : 1e-5f;

    return norm;
}

struct PicoTensor* pico_nn_layernorm_forward(struct PicoContext* ctx, struct PicoLayerNorm* norm,
                                             struct PicoTensor* input) {
    if(ctx == NULL || norm == NULL || input == NULL) {
        return NULL;
    }

    if(input->ndim < 1 || input->shape[input->ndim - 1] != norm->normalized_dim) {
        fprintf(stderr, "PicoNormError: layernorm input last dim must match normalized_dim\n");
        return NULL;
    }

    struct Arena* arena = pico_context_arena(ctx);
    if(arena == NULL) {
        fprintf(stderr, "PicoArenaError: no arena available for layernorm forward allocation\n");
        return NULL;
    }

    int dim = input->ndim - 1;
    struct PicoTensor* mean = pico_mean(ctx, input, dim);
    struct PicoTensor* var = pico_var(ctx, input, dim);
    if(mean == NULL || var == NULL) {
        return NULL;
    }

    int64_t* reduced_shape = arena_alloc(arena, sizeof(int64_t) * input->ndim);
    if(reduced_shape == NULL) {
        return NULL;
    }

    for(int d = 0; d < input->ndim; d++) {
        reduced_shape[d] = d == dim ? 1 : input->shape[d];
    }

    pico_view(ctx, mean, reduced_shape, input->ndim);
    pico_view(ctx, var, reduced_shape, input->ndim);

    struct PicoTensor* centered = pico_sub(ctx, input, mean);
    struct PicoTensor* eps = pico_tensor_from_scalar(ctx, norm->eps);
    struct PicoTensor* denom = pico_sqrt(ctx, pico_add(ctx, var, eps));
    if(centered == NULL || denom == NULL) {
        return NULL;
    }

    return pico_div(ctx, centered, denom);
}

void pico_nn_layernorm_free(struct PicoLayerNorm* norm) {
    if(norm == NULL) {
        return;
    }

    free(norm);
}
