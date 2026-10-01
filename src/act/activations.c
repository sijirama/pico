#include <math.h>
#define _USE_MATH_DEFINES
#include "../arena.h"
#include "../ctx.h"
#include "../kernels/cuda/cuda_ops.h"
#include "../tensor.h"
#include "act_autograd.h"
#include "activations.h"

struct PicoTensor *pico_relu(struct PicoContext *ctx, struct PicoTensor *x) {
    struct Arena *arena = pico_context_arena(ctx);
    if(arena == NULL) {
        fprintf(stderr, "PicoArenaError: no arena available for relu allocation\n");
        return NULL;
    }
    struct PicoTensor *out = pico_create_tensor_on(ctx, x->backend, x->shape, x->ndim);

    if(x->backend == PICO_BACKEND_CPU) {
        for(int i = 0; i < x->numel; i++) {
            out->data[i] = MAX(x->data[i], 0);
        }
    } else if(x->backend == PICO_BACKEND_CUDA) {
        if(!pico_cuda_relu(x, out)) {
            return NULL;
        }
    } else {
        fprintf(stderr, "PicoBackendError: relu unknown backend\n");
        return NULL;
    }

    out->parents = arena_alloc(arena, sizeof(struct PicoTensor *));
    out->parents[0] = x;
    out->num_parents = 1;
    out->_backward = pico_relu_backward;

    return out;
}

struct PicoTensor *pico_sigmoid(struct PicoContext *ctx, struct PicoTensor *x) {
    struct Arena *arena = pico_context_arena(ctx);
    if(arena == NULL) {
        fprintf(stderr, "PicoArenaError: no arena available for sigmoid allocation\n");
        return NULL;
    }
    struct PicoTensor *out = pico_create_tensor_on(ctx, x->backend, x->shape, x->ndim);

    if(x->backend == PICO_BACKEND_CPU) {
        for(int i = 0; i < x->numel; i++) {
            out->data[i] = sigmoid(x->data[i]);
        }
    } else if(x->backend == PICO_BACKEND_CUDA) {
        if(!pico_cuda_sigmoid(x, out)) {
            return NULL;
        }
    } else {
        fprintf(stderr, "PicoBackendError: sigmoid unknown backend\n");
        return NULL;
    }

    out->parents = arena_alloc(arena, sizeof(struct PicoTensor *));
    out->parents[0] = x;
    out->num_parents = 1;
    out->_backward = pico_sigmoid_backward;

    return out;
}

struct PicoTensor *pico_swiglu(struct PicoContext *ctx, struct PicoTensor *x, struct PicoTensor *gate) {
    if(ctx == NULL || x == NULL || gate == NULL) {
        return NULL;
    }

    if(!pico_require_same_backend(x, gate, "swiglu")) {
        return NULL;
    }

    if(!pico_tensor_shapes_are_equal(x, gate)) {
        fprintf(stderr, "PicoActivationError: swiglu inputs must have the same shape\n");
        return NULL;
    }

    struct Arena *arena = pico_context_arena(ctx);
    if(arena == NULL) {
        fprintf(stderr, "PicoArenaError: no arena available for swiglu allocation\n");
        return NULL;
    }

    struct PicoTensor *out = pico_create_tensor_on(ctx, x->backend, x->shape, x->ndim);
    if(out == NULL) {
        return NULL;
    }

    if(x->backend == PICO_BACKEND_CPU) {
        for(int64_t i = 0; i < x->numel; i++) {
            out->data[i] = silu(x->data[i]) * gate->data[i];
        }
    } else if(x->backend == PICO_BACKEND_CUDA) {
        if(!pico_cuda_swiglu(x, gate, out)) {
            return NULL;
        }
    } else {
        fprintf(stderr, "PicoBackendError: swiglu unknown backend\n");
        return NULL;
    }

    out->parents = arena_alloc(arena, sizeof(struct PicoTensor *) * 2);
    if(out->parents == NULL) {
        return NULL;
    }
    out->parents[0] = x;
    out->parents[1] = gate;
    out->num_parents = 2;
    out->_backward = pico_swiglu_backward;

    return out;
}
