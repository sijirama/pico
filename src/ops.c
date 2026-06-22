#include "ops.h"

#include <stdint.h>
#include <stdio.h>

#include "arena.h"
#include "autograd.h"
#include "kernels/cpu_kernels.h"
#include "tensor.h"

struct PicoTensor* pico_add(struct PicoTensor* a, struct PicoTensor* b) {
    if(!pico_check_broadcast_compatibility(a, b)) {
        fprintf(stderr, "[Pico] Error: Shapes are not broadcastable!\n");
        return NULL;
    }

    if(a->backend != b->backend) {
        fprintf(stderr, "[Pico] Error: PicoTensor backends are not compatible!\n");
        return NULL;
    }

    struct Arena* arena = arena_ctx_current();
    if(arena == NULL) {
        fprintf(stderr, "[Pico] Error: No current arena in context!\n");
        return NULL;
    }

    int ndim = MAX(a->ndim, b->ndim);
    int64_t* a_padded_shape = pad_shape(arena, a, ndim);
    int64_t* b_padded_shape = pad_shape(arena, b, ndim);

    int64_t* res_shape = arena_alloc(arena, sizeof(int64_t) * ndim);
    for(int i = 0; i < ndim; i++)
        res_shape[i] = MAX(a_padded_shape[i], b_padded_shape[i]);

    struct PicoTensor* out = pico_create_tensor(arena, res_shape, ndim);
    out->backend = a->backend;

    if(a->backend == CPU) {
        pico_add_cpu(a, b, out);
    } else if(a->backend == GPU) {
        // pico_add_gpu(a, b, out);
    }

    // stuff we need for backprop
    out->parents = arena_alloc(arena, sizeof(struct PicoTensor*) * 2);
    out->parents[0] = a;
    out->parents[1] = b;
    out->num_parents = 2;
    out->_backward = pico_add_backward;

    return out;
}

struct PicoTensor* pico_sub(struct PicoTensor* a, struct PicoTensor* b) {
    if(!pico_check_broadcast_compatibility(a, b)) {
        fprintf(stderr, "[Pico] Error: Shapes are not broadcastable!\n");
        return NULL;
    }

    if(a->backend != b->backend) {
        fprintf(stderr, "[Pico] Error: PicoTensor backends are not compatible!\n");
        return NULL;
    }

    struct Arena* arena = arena_ctx_current();
    if(arena == NULL) {
        fprintf(stderr, "[Pico] Error: No current arena in context!\n");
        return NULL;
    }

    int ndim = MAX(a->ndim, b->ndim);
    int64_t* a_padded_shape = pad_shape(arena, a, ndim);
    int64_t* b_padded_shape = pad_shape(arena, b, ndim);

    int64_t* res_shape = arena_alloc(arena, sizeof(int64_t) * ndim);
    for(int i = 0; i < ndim; i++)
        res_shape[i] = MAX(a_padded_shape[i], b_padded_shape[i]);

    struct PicoTensor* out = pico_create_tensor(arena, res_shape, ndim);
    out->backend = a->backend;

    if(a->backend == CPU) {
        pico_sub_cpu(a, b, out);
    } else if(a->backend == GPU) {
        // pico_sub_gpu(a, b, out);
    }

    // stuff we need for backprop
    out->parents = arena_alloc(arena, sizeof(struct PicoTensor*) * 2);
    out->parents[0] = a;
    out->parents[1] = b;
    out->num_parents = 2;
    out->_backward = pico_sub_backward;

    return out;
}
