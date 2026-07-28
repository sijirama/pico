#include "linear.h"

#include <stdbool.h>
#include <stdlib.h>

#include "arena.h"
#include "ctx.h"
#include "ops.h"
#include "tensor.h"

struct PicoLinear* pico_nn_linear_init(struct PicoContext* ctx, int in_features, int out_features, bool bias) {
    struct Arena* arena = pico_context_arena(ctx);
    if(arena == NULL) {
        fprintf(stderr, "PicoArenaError: no arena available for linear init allocation\n");
        return NULL;
    }

    int64_t* res_shape = arena_alloc(arena, sizeof(int64_t) * 2);
    res_shape[0] = in_features;
    res_shape[1] = out_features;

    int64_t* bias_shape = arena_alloc(arena, sizeof(int64_t) * 2);
    bias_shape[0] = out_features;

    struct PicoTensor* weights_t = pico_param(ctx, res_shape, 2);
    struct PicoTensor* bias_t = NULL;

    if(bias == true) {
        bias_t = pico_param(ctx, bias_shape, 1);
    }

    struct PicoLinear* linear = malloc(sizeof(struct PicoLinear));

    linear->weights = weights_t;
    linear->bias = bias_t;
    linear->in_features = in_features;
    linear->out_features = out_features;

    return linear;
}

struct PicoTensor* pico_nn_linear_forward(struct PicoContext* ctx, struct PicoLinear* layer, struct PicoTensor* input) {
    //
    //
    //
    if(input->shape[input->ndim - 1] != layer->weights->shape[0]) {
        perror("[Pico] Error:  In Linear - 2 matmuls matrices must be compatible");
        return NULL;
    }

    if(layer->weights->backend != input->backend) {
        fprintf(stderr, "[Pico] Error: In Linear - PicoTensor backends are not compatible!\n");
        return NULL;
    }

    if(pico_context_arena(ctx) == NULL) {
        fprintf(stderr, "PicoArenaError: no arena available for linear forward allocation\n");
        return NULL;
    }

    struct PicoTensor* output = pico_matmul(ctx, input, layer->weights);

    if(layer->bias != NULL) {
        output = pico_add(ctx, output, layer->bias);
    }

    return output;
}

void pico_nn_linear_free(struct PicoLinear* linear) {
    if(linear == NULL) {
        return;
    }

    free(linear);
}
