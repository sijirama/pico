#include "tensor.h"

#include <math.h>
#include <sched.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "arena.h"
#include "ctx.h"
#include "global.h"
#include "kernels/cuda/cuda_ops.h"
#include "lib/pico_vector.h"
#include "ops.h"

void postorder(
    struct PicoTensor *root,
    struct PicoVec *vector,
    struct PicoVec *visited);

// INFO: backward walks the graph from the output back to
// leaves. the temporary vectors are just traversal scratch,
// so they stay outside tensor ownership rules.
void pico_backward(
    struct PicoContext *ctx, struct PicoTensor *entry) {
    (void)ctx;

    // build our dependency graph with dfs
    struct PicoVec vector, visited;
    pico_vec_init(&vector, 25);
    pico_vec_init(&visited, 25);
    postorder(entry, &vector, &visited);

    // post-order gives [leaves ... entry]; reverse ->
    // [entry ... leaves]
    pico_vec_reverse(&vector);

    // seed the entry node with grad 1
    struct PicoTensor *curr = NULL;
    curr = (struct PicoTensor *)vector.data[0];
    if(curr->backend == PICO_BACKEND_CUDA) {
        pico_cuda_fill(curr, 1.0f);
    } else {
        for(int i = 0; i < curr->numel; i++) {
            curr->grad[i] = 1.0f;
        }
    }

    // call backward on each  (now iterate FORWARD: entry is
    // first)
    for(int i = 0; i < vector.size; i++) {
        curr = (struct PicoTensor *)vector.data[i];
        if(curr->_backward != NULL) {
            curr->_backward(curr);
        }
    }

    pico_vec_free(&vector);
    pico_vec_free(&visited);
}

static struct PicoTensor *pico_tensor_alloc_metadata(
    struct Arena *arena,
    int64_t *shape,
    uint8_t ndim,
    enum PicoTensorKind kind,
    PicoBackend backend) {
    if(arena == NULL) {
        return NULL;
    }

    struct PicoTensor *tensor =
        (struct PicoTensor *)arena_alloc(arena, sizeof(struct PicoTensor));
    if(tensor == NULL) {
        return NULL;
    }

    memset(tensor, 0, sizeof(struct PicoTensor));
    tensor->ndim = ndim;
    tensor->kind = kind;
    tensor->backend = backend;

    tensor->shape = (int64_t *)arena_alloc(arena, ndim * sizeof(int64_t));
    tensor->strides = (int64_t *)arena_alloc(arena, ndim * sizeof(int64_t));
    if(tensor->shape == NULL || tensor->strides == NULL) {
        return NULL;
    }

    memcpy(tensor->shape, shape, ndim * sizeof(int64_t));
    tensor->numel = pico_compute_numel(tensor->shape, tensor->ndim);
    pico_compute_strides(tensor->shape, tensor->ndim, tensor->strides);

    return tensor;
}

// INFO: params are trainable ctx-owned tensors. metadata lives in param_arena;
// CPU data/grad live there too. CUDA data/grad will live in VRAM later.
struct PicoTensor *pico_param(
    struct PicoContext *ctx, int64_t *shape, uint8_t ndim) {
    return pico_param_named_on(ctx, PICO_BACKEND_CPU, NULL, shape, ndim);
}

struct PicoTensor *pico_param_named(
    struct PicoContext *ctx,
    char *name,
    int64_t *shape,
    uint8_t ndim) {
    return pico_param_named_on(ctx, PICO_BACKEND_CPU, name, shape, ndim);
}

struct PicoTensor *pico_param_on(
    struct PicoContext *ctx, PicoBackend backend, int64_t *shape, uint8_t ndim) {
    return pico_param_named_on(ctx, backend, NULL, shape, ndim);
}

struct PicoTensor *pico_param_named_on(
    struct PicoContext *ctx,
    PicoBackend backend,
    char *name,
    int64_t *shape,
    uint8_t ndim) {
    struct Arena *arena = pico_context_param_arena(ctx);
    if(arena == NULL) {
        fprintf(
            stderr,
            "PicoArenaError: no param arena available for param allocation\n");
        return NULL;
    }

    struct PicoTensor *tensor =
        pico_tensor_alloc_metadata(arena, shape, ndim, PICO_TENSOR_PARAM, backend);
    if(tensor == NULL) {
        return NULL;
    }

    if(name != NULL) {
        tensor->name = arena_alloc(arena, strlen(name) + 1);
        if(tensor->name == NULL) {
            return NULL;
        }
        strcpy(tensor->name, name);
    }

    if(!pico_tensor_init_data_on(ctx, tensor, backend)) {
        return NULL;
    }

    pico_context_register_param(ctx, tensor);

    return tensor;
}

// INFO: temp tensors are arena-backed. this is what ops use
// for outputs and intermediate graph nodes, so a training
// loop can drop them all with one reset.
struct PicoTensor *pico_create_tensor(
    struct PicoContext *ctx, int64_t *shape, uint8_t ndim) {
    return pico_create_tensor_on(ctx, PICO_BACKEND_CPU, shape, ndim);
}

struct PicoTensor *pico_create_tensor_on(
    struct PicoContext *ctx, PicoBackend backend, int64_t *shape, uint8_t ndim) {
    struct Arena *arena = pico_context_temp_arena(ctx);
    if(arena == NULL) {
        fprintf(
            stderr,
            "PicoArenaError: no temp arena available for tensor "
            "allocation\n");
        return NULL;
    }

    struct PicoTensor *tensor =
        pico_tensor_alloc_metadata(arena, shape, ndim, PICO_TENSOR_TEMP, backend);
    if(tensor == NULL) {
        return NULL;
    }

    if(!pico_tensor_init_data_on(ctx, tensor, backend)) {
        return NULL;
    }

    return tensor;
}

// a 1-element tensor holding `value`. shape {1} ->
// broadcasts against anything via map_index (the size-1 dim
// is stretched). leaf tensor: no parents, _backward NULL
// (pico_create_tensor already sets those), so it acts as a
// constant in the graph.
struct PicoTensor *pico_tensor_from_scalar(
    struct PicoContext *ctx, float value) {
    return pico_tensor_from_scalar_on(ctx, PICO_BACKEND_CPU, value);
}

struct PicoTensor *pico_tensor_from_scalar_on(
    struct PicoContext *ctx, PicoBackend backend, float value) {
    int64_t shape[1] = {1};
    struct PicoTensor *tensor =
        pico_create_tensor_on(ctx, PICO_BACKEND_CPU, shape, 1);
    if(tensor == NULL) {
        return NULL;
    }
    tensor->data[0] = value;

    if(!pico_tensor_to_backend(ctx, tensor, backend)) {
        return NULL;
    }

    return tensor;
}

// INFO: this is a copy constructor for temp data. it is
// intentionally not a view into the caller's array, because
// pico cannot know how long that pointer lives.
struct PicoTensor *pico_tensor_from_data(
    struct PicoContext *ctx,
    int64_t *shape,
    uint8_t ndim,
    const float *data) {
    return pico_tensor_from_data_on(ctx, PICO_BACKEND_CPU, shape, ndim, data);
}

struct PicoTensor *pico_tensor_from_data_on(
    struct PicoContext *ctx,
    PicoBackend backend,
    int64_t *shape,
    uint8_t ndim,
    const float *data) {
    if(data == NULL) {
        fprintf(
            stderr,
            "[Pico] Error: pico_tensor_from_data received "
            "NULL data\n");
        return NULL;
    }

    struct PicoTensor *tensor =
        pico_create_tensor_on(ctx, PICO_BACKEND_CPU, shape, ndim);
    if(tensor == NULL) {
        return NULL;
    }

    memcpy(
        tensor->data, data, tensor->numel * sizeof(float));

    if(!pico_tensor_to_backend(ctx, tensor, backend)) {
        return NULL;
    }

    return tensor;
}

// recursive helper: walk one dim, indent nested brackets,
// use strides so a non-contiguous / broadcasted view still
// prints in logical shape order.
static void pico_print_recursive(
    struct PicoTensor *t, int dim, int64_t offset) {
    if(dim ==
       t->ndim - 1) { // innermost axis -> print the row
        printf("[");
        for(int64_t i = 0; i < t->shape[dim]; i++) {
            printf(
                "%g",
                t->data[offset + i * t->strides[dim]]);
            if(i != t->shape[dim] - 1)
                printf(", ");
        }
        printf("]");
        return;
    }
    printf("[");
    for(int64_t i = 0; i < t->shape[dim]; i++) {
        pico_print_recursive(
            t, dim + 1, offset + i * t->strides[dim]);
        if(i != t->shape[dim] - 1)
            printf(",\n ");
    }
    printf("]");
}

void pico_tensor_print(struct PicoTensor *t) {
    if(t == NULL) {
        printf("PicoTensor(NULL)\n");
        return;
    }
    printf("PicoTensor(shape=[");
    for(int i = 0; i < t->ndim; i++) {
        printf("%ld", (long)t->shape[i]);
        if(i != t->ndim - 1)
            printf(", ");
    }
    printf("], numel=%ld)\n", (long)t->numel);

    if(t->ndim == 0 || t->data == NULL) {
        printf("(no data)\n");
        return;
    }
    pico_print_recursive(t, 0, 0);
    printf("\n");
}

// ============================= pico_rand

// Fast Xorshift32 generator
static inline uint32_t xorshift32(void) {
    uint32_t x = x_state;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    return x_state = x;
}

// Bulk generate floats directly via IEEE-754 bit-casting
void generate_random_floats_fast(float *arr, size_t size) {
    for(size_t i = 0; i < size; i++) {
        uint32_t r = xorshift32();
        // Construct a float in the range [1.0, 2.0) by
        // setting mantissa bits
        uint32_t bits = (r >> 9) | 0x3F800000;
        float f = *(float *)&bits;
        arr[i] = f - 1.0f; // Shift range down to [0.0, 1.0)
    }
}

// INFO: rand returns a temp tensor. pass an explicit arena
// for short-lived random scratch, or NULL after pico_init
// if the default arena is enough.
struct PicoTensor *pico_rand(
    struct PicoContext *ctx, int64_t *shape, uint8_t ndim) {
    struct PicoTensor *tensor =
        pico_create_tensor(ctx, shape, ndim);
    // dispatch properly into backends
    generate_random_floats_fast(
        tensor->data, tensor->numel);
    return tensor;
}

// ============================= pico_randn

// INFO: Box-Muller gives us two normal samples from two
// uniform samples. generate by flat numel, then write into
// a tensor with the original requested shape so odd sizes
// and multidim shapes don't come back with weird metadata.
// WARN: this code was written with ai lmao
struct PicoTensor *pico_randn(
    struct PicoContext *ctx, int64_t *shape, uint8_t ndim) {
    struct Arena *arena = pico_context_arena(ctx);
    if(arena == NULL) {
        fprintf(
            stderr,
            "PicoArenaError: no arena available for randn "
            "allocation\n");
        return NULL;
    }

    struct PicoTensor *tensor =
        pico_create_tensor(ctx, shape, ndim);
    if(tensor == NULL) {
        return NULL;
    }

    for(int64_t i = 0; i < tensor->numel; i += 2) {
        float u1 = 0.0f;
        generate_random_floats_fast(&u1, 1);
        if(u1 == 0.0f) {
            u1 = 1.0f / 16777216.0f;
        }

        float u2 = 0.0f;
        generate_random_floats_fast(&u2, 1);

        float mag = sqrtf(-2.0f * logf(u1));
        float angle = 2.0f * PI_F * u2;

        tensor->data[i] = mag * cosf(angle);
        if(i + 1 < tensor->numel) {
            tensor->data[i + 1] = mag * sinf(angle);
        }
    }

    return tensor;
}

// ============================= end

uint8_t pico_check_broadcast_compatibility(
    struct PicoTensor *a, struct PicoTensor *b) {
    int ndim_a = a->ndim;
    int ndim_b = b->ndim;

    // We check from the end of the shape arrays (the
    // "trailing" dimensions)
    int i = ndim_a - 1;
    int j = ndim_b - 1;

    while(i >= 0 && j >= 0) {
        int dim_a = a->shape[i];
        int dim_b = b->shape[j];

        // The Broadcasting Rule:
        // 1. Dimensions are equal, OR
        // 2. One of them is 1
        if(dim_a != dim_b && dim_a != 1 && dim_b != 1) {
            return 0; // Not compatible!
        }
        i--;
        j--;
    }

    // If one tensor has more dimensions (e.g., [5, 4, 3] vs
    // [4, 3]), the extra leading dimensions [5] are always
    // compatible with the "implicit ones" of the smaller
    // tensor.
    return 1;
}

void postorder(
    struct PicoTensor *root,
    struct PicoVec *vector,
    struct PicoVec *visited) {
    if(root == NULL) {
        return;
    }
    if(pico_vec_find(visited, root) !=
       -1) { // if node was found? stop redundant traversals
        return;
    }

    pico_vec_push(visited, root);

    for(int i = 0; i < root->num_parents; i++) {
        postorder(root->parents[i], vector, visited);
    }

    // append to array if not appended before
    pico_vec_push(vector, root);
}
