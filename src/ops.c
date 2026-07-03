#include "ops.h"

#include <stdint.h>
#include <stdio.h>
#include <string.h>

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

struct PicoTensor* pico_mul(struct PicoTensor* a, struct PicoTensor* b) {
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

struct PicoTensor* pico_matmul(struct PicoTensor* a, struct PicoTensor* b) {
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

    struct Arena* arena = arena_ctx_current();
    if(arena == NULL) {
        fprintf(stderr, "[Pico] Error: No current arena in context!\n");
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

struct PicoTensor* pico_cat(struct PicoTensor* a, struct PicoTensor* b, int dim) {
    if(a->backend != b->backend) {
        fprintf(
            stderr,
            "[Pico] Error: PicoTensor backends are not compatible, Mismatch found in backends!\n");
        return NULL;
    }
    if(a->ndim != b->ndim) {
        fprintf(stderr,
                "[Pico] Error: PicoTensors are not compatible for contatenation, Mismatch found in "
                "ndim!\n");
        return NULL;
    }

    struct Arena* arena = arena_ctx_current();
    if(arena == NULL) {
        fprintf(stderr, "[Pico] Error: No current arena in context!\n");
        return NULL;
    }

    int64_t* res_shape = arena_alloc(arena, sizeof(int64_t) * a->ndim);

    // dim=0 means stack them over each other dim=1 means side by side
    for(int i = 0; i < a->ndim; i++) {
        if(i == dim) {
            res_shape[i] = a->shape[i] + b->shape[i];
            continue;
        }
        if(a->shape[i] != b->shape[i]) {
            fprintf(stderr,
                    "[Pico] Error: PicoTensors are not compatible for contatenation, Mismatch "
                    "found in shape!\n");
            return NULL;
        }
        res_shape[i] = a->shape[i];
    }

    struct PicoTensor* out = pico_create_tensor(arena, res_shape, a->ndim);

    float* src_a = (float*)a->data;
    float* src_b = (float*)b->data;
    float* dst = (float*)out->data;

    int64_t outer_count = 1;
    for(int i = 0; i < dim; i++) {
        outer_count *= a->shape[i];
    }

    int64_t inner_size = 1;
    for(int i = dim + 1; i < a->ndim; i++) {
        inner_size *= a->shape[i];
    }

    int64_t a_copy_size = a->shape[dim] * inner_size;
    int64_t b_copy_size = b->shape[dim] * inner_size;

    for(int64_t o = 0; o < outer_count; o++) {
        // 1. Copy chunk from tensor A
        memcpy(dst, src_a, a_copy_size * sizeof(float));
        dst += a_copy_size;
        src_a += a_copy_size;

        // 2. Copy chunk from tensor B right next to it
        memcpy(dst, src_b, b_copy_size * sizeof(float));
        dst += b_copy_size;
        src_b += b_copy_size;
    }

    return out;
}

// ---- unary element-wise math (forward only) -------------------------------
// same shape as `out`, dispatch to the CPU kernel, wire the single parent so the
// graph stays intact. _backward is NULL for now — the per-op backwards are TODO
// (sin'=cos, cos'=-sin, tan'=sec^2, tanh'=1-tanh^2, sqrt'=1/(2*sqrt)). unary =>
// num_parents == 1. these five are near-identical: prime for a later bundle.

struct PicoTensor* pico_tensor_sqrt(struct PicoTensor* a) {
    struct Arena* arena = arena_ctx_current();
    if(arena == NULL) {
        fprintf(stderr, "[Pico] Error: No current arena in context!\n");
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
    out->_backward = NULL;  // TODO: sqrt backward

    return out;
}

struct PicoTensor* pico_tensor_sin(struct PicoTensor* a) {
    struct Arena* arena = arena_ctx_current();
    if(arena == NULL) {
        fprintf(stderr, "[Pico] Error: No current arena in context!\n");
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
    out->_backward = NULL;  // TODO: sin backward (cos)

    return out;
}

struct PicoTensor* pico_tensor_cos(struct PicoTensor* a) {
    struct Arena* arena = arena_ctx_current();
    if(arena == NULL) {
        fprintf(stderr, "[Pico] Error: No current arena in context!\n");
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
    out->_backward = NULL;  // TODO: cos backward (-sin)

    return out;
}

struct PicoTensor* pico_tensor_tan(struct PicoTensor* a) {
    struct Arena* arena = arena_ctx_current();
    if(arena == NULL) {
        fprintf(stderr, "[Pico] Error: No current arena in context!\n");
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
    out->_backward = NULL;  // TODO: tan backward (sec^2)

    return out;
}

struct PicoTensor* pico_tensor_tanh(struct PicoTensor* a) {
    struct Arena* arena = arena_ctx_current();
    if(arena == NULL) {
        fprintf(stderr, "[Pico] Error: No current arena in context!\n");
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
    out->_backward = NULL;  // TODO: tanh backward (1 - tanh^2)

    return out;
}

struct PicoTensor* pico_tensor_log(struct PicoTensor* a) {
    struct Arena* arena = arena_ctx_current();
    if(arena == NULL) {
        fprintf(stderr, "[Pico] Error: No current arena in context!\n");
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
    out->_backward = NULL;  // TODO: log backward (1/x)

    return out;
}
