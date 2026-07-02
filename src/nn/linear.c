#include "linear.h"

#include <stdbool.h>

#include "arena.h"
#include "ops.h"
#include "tensor.h"

struct PicoLinear* pico_nn_linear_init(int in_features, int out_features, bool bias) {
    struct Arena* arena = arena_ctx_current();
    if(arena == NULL) {
        fprintf(stderr, "[Pico] Error: No current arena in context!\n");
        return NULL;
    }

    int64_t* res_shape = arena_alloc(arena, sizeof(int64_t) * 2);
    res_shape[0] = in_features;
    res_shape[1] = out_features;

    int64_t* bias_shape = arena_alloc(arena, sizeof(int64_t) * 2);
    bias_shape[0] = out_features;

    struct PicoTensor* weights_t = pico_param(res_shape, 2);
    struct PicoTensor* bias_t = NULL;

    if(bias == true) {
        bias_t = pico_param(bias_shape, 1);
    }

    struct PicoLinear* linear = malloc(sizeof(struct PicoLinear));

    linear->weights = weights_t;
    linear->bias = bias_t;
    linear->in_features = in_features;
    linear->out_features = out_features;

    return linear;
}

struct PicoTensor* pico_nn_linear_forward(struct PicoLinear* layer, struct PicoTensor* input) {
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

    struct Arena* arena = arena_ctx_current();
    if(arena == NULL) {
        fprintf(stderr, "[Pico] Error: In Linear - No current arena in context!\n");
        return NULL;
    }

    struct PicoTensor* output = pico_matmul(input, layer->weights);

    if(layer->bias != NULL) {
        output = pico_add(output, layer->bias);
    }

    return output;
}

void pico_nn_linear_free(struct PicoLinear* linear) {
    if(linear == NULL) {
        return;
    }

    pico_free(linear->weights);
    pico_free(linear->bias);
    free(linear);
}
