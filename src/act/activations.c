#include <math.h>
#define _USE_MATH_DEFINES
#include "activations.h"
#include "arena.h"
#include "act_autograd.h"
#include "ctx.h"
#include "tensor.h"

struct PicoTensor* pico_relu(struct PicoContext* ctx, struct PicoTensor* x) {
    if(!pico_require_cpu_backend(x->backend, "relu")) {
        return NULL;
    }

    struct Arena* arena = pico_context_arena(ctx);
    if(arena == NULL) {
        fprintf(stderr, "PicoArenaError: no arena available for relu allocation\n");
        return NULL;
    }
    struct PicoTensor* out = pico_create_tensor_on(ctx, x->backend, x->shape, x->ndim);

    for(int i = 0; i < x->numel; i++) {
        out->data[i] = MAX(x->data[i], 0);
    }

    out->parents = arena_alloc(arena, sizeof(struct PicoTensor*));
    out->parents[0] = x;
    out->num_parents = 1;
    out->_backward = pico_relu_backward;

    return out;
}

struct PicoTensor* pico_sigmoid(struct PicoContext* ctx, struct PicoTensor* x) {
    if(!pico_require_cpu_backend(x->backend, "sigmoid")) {
        return NULL;
    }

    struct Arena* arena = pico_context_arena(ctx);
    if(arena == NULL) {
        fprintf(stderr, "PicoArenaError: no arena available for sigmoid allocation\n");
        return NULL;
    }
    struct PicoTensor* out = pico_create_tensor_on(ctx, x->backend, x->shape, x->ndim);

    for(int i = 0; i < x->numel; i++) {
        out->data[i] = sigmoid(x->data[i]);
    }

    out->parents = arena_alloc(arena, sizeof(struct PicoTensor*));
    out->parents[0] = x;
    out->num_parents = 1;
    out->_backward = pico_sigmoid_backward;

    return out;
}
