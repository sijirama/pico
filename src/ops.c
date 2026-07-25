#include "ops.h"

#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include "arena.h"
#include "autograd.h"
#include "kernels/cpu_kernels.h"
#include "tensor.h"

struct PicoTensor* pico_add(struct Arena* arena, struct PicoTensor* a, struct PicoTensor* b) {
    if(!pico_check_broadcast_compatibility(a, b)) {
        fprintf(stderr, "[Pico] Error: Shapes are not broadcastable!\n");
        return NULL;
    }

    if(a->backend != b->backend) {
        fprintf(stderr, "[Pico] Error: PicoTensor backends are not compatible!\n");
        return NULL;
    }

    arena = arena_resolve(arena);
    if(arena == NULL) {
        fprintf(stderr, "PicoArenaError: no arena available for add allocation\n");
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

struct PicoTensor* pico_sub(struct Arena* arena, struct PicoTensor* a, struct PicoTensor* b) {
    if(!pico_check_broadcast_compatibility(a, b)) {
        fprintf(stderr, "[Pico] Error: Shapes are not broadcastable!\n");
        return NULL;
    }

    if(a->backend != b->backend) {
        fprintf(stderr, "[Pico] Error: PicoTensor backends are not compatible!\n");
        return NULL;
    }

    arena = arena_resolve(arena);
    if(arena == NULL) {
        fprintf(stderr, "PicoArenaError: no arena available for sub allocation\n");
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

struct PicoTensor* pico_mul(struct Arena* arena, struct PicoTensor* a, struct PicoTensor* b) {
    if(!pico_check_broadcast_compatibility(a, b)) {
        fprintf(stderr, "[Pico] Error: Shapes are not broadcastable!\n");
        return NULL;
    }

    if(a->backend != b->backend) {
        fprintf(stderr, "[Pico] Error: PicoTensor backends are not compatible!\n");
        return NULL;
    }

    arena = arena_resolve(arena);
    if(arena == NULL) {
        fprintf(stderr, "PicoArenaError: no arena available for mul allocation\n");
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
        pico_mul_cpu(a, b, out);
    } else if(a->backend == GPU) {
        // pico_sub_gpu(a, b, out);
    }

    // stuff we need for backprop
    out->parents = arena_alloc(arena, sizeof(struct PicoTensor*) * 2);
    out->parents[0] = a;
    out->parents[1] = b;
    out->num_parents = 2;
    out->_backward = pico_mul_backward;

    return out;
}

struct PicoTensor* pico_matmul(struct Arena* arena, struct PicoTensor* a, struct PicoTensor* b) {
    if(a->shape[a->ndim - 1] != b->shape[0]) {
        perror("[Pico] Error: 2 matmuls matrices must be compatible");
        return NULL;
    }

    if(a->ndim != 2 || b->ndim != 2) {
        perror("[Pico] Error: 2d matmul matrices must be compatible");
        return NULL;
    }

    if(a->backend != b->backend) {
        fprintf(stderr, "[Pico] Error: PicoTensor backends are not compatible!\n");
        return NULL;
    }

    arena = arena_resolve(arena);
    if(arena == NULL) {
        fprintf(stderr, "PicoArenaError: no arena available for matmul allocation\n");
        return NULL;
    }

    int ndim = MAX(a->ndim, b->ndim);
    int rows = a->shape[0];
    int columns = b->shape[1];

    int64_t* res_shape = arena_alloc(arena, sizeof(int64_t) * ndim);
    res_shape[0] = rows;
    res_shape[1] = columns;

    struct PicoTensor* out = pico_create_tensor(arena, res_shape, ndim);
    out->backend = a->backend;  // new tensor backend is consistent with it's parents, born in the
                                // same fucking realm

    if(a->backend == CPU) {
        pico_matmul_cpu(a, b, out);
    }

    // stuff we need for backprop
    out->parents = arena_alloc(arena, sizeof(struct PicoTensor*) * 2);
    out->parents[0] = a;
    out->parents[1] = b;
    out->num_parents = 2;
    out->_backward = pico_matmul_backward;

    return out;
}

// ---- unary element-wise math ----------------------------------------------
// same shape as `out`, dispatch to the CPU kernel, wire the single parent so the
// graph stays intact. unary => num_parents == 1. these are near-identical:
// prime for a later bundle.

struct PicoTensor* pico_sqrt(struct Arena* arena, struct PicoTensor* a) {
    arena = arena_resolve(arena);
    if(arena == NULL) {
        fprintf(stderr, "PicoArenaError: no arena available for sqrt allocation\n");
        return NULL;
    }

    struct PicoTensor* out = pico_create_tensor(arena, a->shape, a->ndim);
    out->backend = a->backend;

    if(a->backend == CPU) {
        pico_sqrt_cpu(a, out);
    }

    out->parents = arena_alloc(arena, sizeof(struct PicoTensor*));
    out->parents[0] = a;
    out->num_parents = 1;
    out->_backward = pico_sqrt_backward;

    return out;
}

struct PicoTensor* pico_sin(struct Arena* arena, struct PicoTensor* a) {
    arena = arena_resolve(arena);
    if(arena == NULL) {
        fprintf(stderr, "PicoArenaError: no arena available for sin allocation\n");
        return NULL;
    }

    struct PicoTensor* out = pico_create_tensor(arena, a->shape, a->ndim);
    out->backend = a->backend;

    if(a->backend == CPU) {
        pico_sin_cpu(a, out);
    }

    out->parents = arena_alloc(arena, sizeof(struct PicoTensor*));
    out->parents[0] = a;
    out->num_parents = 1;
    out->_backward = pico_sin_backward;

    return out;
}

struct PicoTensor* pico_cos(struct Arena* arena, struct PicoTensor* a) {
    arena = arena_resolve(arena);
    if(arena == NULL) {
        fprintf(stderr, "PicoArenaError: no arena available for cos allocation\n");
        return NULL;
    }

    struct PicoTensor* out = pico_create_tensor(arena, a->shape, a->ndim);
    out->backend = a->backend;

    if(a->backend == CPU) {
        pico_cos_cpu(a, out);
    }

    out->parents = arena_alloc(arena, sizeof(struct PicoTensor*));
    out->parents[0] = a;
    out->num_parents = 1;
    out->_backward = pico_cos_backward;

    return out;
}

struct PicoTensor* pico_tan(struct Arena* arena, struct PicoTensor* a) {
    arena = arena_resolve(arena);
    if(arena == NULL) {
        fprintf(stderr, "PicoArenaError: no arena available for tan allocation\n");
        return NULL;
    }

    struct PicoTensor* out = pico_create_tensor(arena, a->shape, a->ndim);
    out->backend = a->backend;

    if(a->backend == CPU) {
        pico_tan_cpu(a, out);
    }

    out->parents = arena_alloc(arena, sizeof(struct PicoTensor*));
    out->parents[0] = a;
    out->num_parents = 1;
    out->_backward = pico_tan_backward;

    return out;
}

struct PicoTensor* pico_tanh(struct Arena* arena, struct PicoTensor* a) {
    arena = arena_resolve(arena);
    if(arena == NULL) {
        fprintf(stderr, "PicoArenaError: no arena available for tanh allocation\n");
        return NULL;
    }

    struct PicoTensor* out = pico_create_tensor(arena, a->shape, a->ndim);
    out->backend = a->backend;

    if(a->backend == CPU) {
        pico_tanh_cpu(a, out);
    }

    out->parents = arena_alloc(arena, sizeof(struct PicoTensor*));
    out->parents[0] = a;
    out->num_parents = 1;
    out->_backward = pico_tanh_backward;

    return out;
}

struct PicoTensor* pico_log(struct Arena* arena, struct PicoTensor* a) {
    arena = arena_resolve(arena);
    if(arena == NULL) {
        fprintf(stderr, "PicoArenaError: no arena available for log allocation\n");
        return NULL;
    }

    struct PicoTensor* out = pico_create_tensor(arena, a->shape, a->ndim);
    out->backend = a->backend;

    if(a->backend == CPU) {
        pico_log_cpu(a, out);
    }

    out->parents = arena_alloc(arena, sizeof(struct PicoTensor*));
    out->parents[0] = a;
    out->num_parents = 1;
    out->_backward = pico_log_backward;

    return out;
}
