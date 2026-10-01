#include "linear.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include "../act/activations.h"
#include "../arena.h"
#include "../ctx.h"
#include "../kernels/cuda/cuda_ops.h"
#include "../tensor.h"

static void pico_nn_fused_swiglu_backward(struct PicoTensor* self) {
    struct PicoTensor* gate = self->parents[0];
    struct PicoTensor* up = self->parents[1];
    struct PicoTensor* mask = self->parents[2];

    if(self->backend == PICO_BACKEND_CUDA) {
        pico_cuda_fused_swiglu_backward(self, gate, up, mask);
        return;
    }

    for(int64_t i = 0; i < self->numel; i++) {
        float sig = sigmoid(gate->data[i]);
        float silu_gate = gate->data[i] * sig;
        float silu_grad = sig + gate->data[i] * sig * (1.0f - sig);
        float upstream = self->grad[i] * mask->data[i];

        gate->grad[i] += upstream * up->data[i] * silu_grad;
        up->grad[i] += upstream * silu_gate;
    }
}

struct PicoTensor* pico_nn_fused_swiglu(struct PicoContext* ctx, struct PicoTensor* gate, struct PicoTensor* up,
                                        float dropout_p) {
    if(ctx == NULL || gate == NULL || up == NULL) {
        return NULL;
    }

    if(dropout_p < 0.0f || dropout_p >= 1.0f) {
        fprintf(stderr, "PicoFFNError: fused_swiglu dropout_p must be in [0, 1)\n");
        return NULL;
    }

    if(!pico_require_same_backend(gate, up, "fused_swiglu")) {
        return NULL;
    }

    if(!pico_tensor_shapes_are_equal(gate, up)) {
        fprintf(stderr, "PicoFFNError: fused_swiglu inputs must have the same shape\n");
        return NULL;
    }

    struct Arena* arena = pico_context_arena(ctx);
    if(arena == NULL) {
        fprintf(stderr, "PicoArenaError: no arena available for fused_swiglu allocation\n");
        return NULL;
    }

    struct PicoTensor* out = pico_create_tensor_on(ctx, gate->backend, gate->shape, gate->ndim);
    if(out == NULL) {
        return NULL;
    }

    struct PicoTensor* mask = pico_create_tensor_on(ctx, gate->backend, gate->shape, gate->ndim);
    if(mask == NULL) {
        return NULL;
    }

    if(gate->backend == PICO_BACKEND_CPU) {
        if(ctx->mode == PICO_EVAL || dropout_p == 0.0f) {
            for(int64_t i = 0; i < gate->numel; i++) {
                mask->data[i] = 1.0f;
                out->data[i] = silu(gate->data[i]) * up->data[i];
            }
        } else {
            struct PicoTensor* random = pico_rand(ctx, gate->shape, gate->ndim);
            if(random == NULL) {
                return NULL;
            }

            float scale = 1.0f / (1.0f - dropout_p);
            for(int64_t i = 0; i < gate->numel; i++) {
                mask->data[i] = random->data[i] >= dropout_p ? scale : 0.0f;
                out->data[i] = silu(gate->data[i]) * up->data[i] * mask->data[i];
            }
        }
    } else if(gate->backend == PICO_BACKEND_CUDA) {
        uint32_t seed = (uint32_t)rand();
        if(!pico_cuda_fused_swiglu(gate, up, mask, out, dropout_p, ctx->mode == PICO_TRAIN, seed)) {
            return NULL;
        }
    } else {
        fprintf(stderr, "PicoBackendError: fused_swiglu unknown backend\n");
        return NULL;
    }

    out->parents = arena_alloc(arena, sizeof(struct PicoTensor*) * 3);
    if(out->parents == NULL) {
        return NULL;
    }
    out->parents[0] = gate;
    out->parents[1] = up;
    out->parents[2] = mask;
    out->num_parents = 3;
    out->_backward = pico_nn_fused_swiglu_backward;

    return out;
}
