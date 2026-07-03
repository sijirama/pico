#include "tensor.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "arena.h"
#include "global.h"
#include "lib/pico_vector.h"
#include "ops.h"

void postorder(struct PicoTensor* root, struct PicoVec* vector, struct PicoVec* visited);

void pico_backward(struct Arena* arena, struct PicoTensor* entry) {
    // build our dependency graph with dfs
    struct PicoVec vector, visited;
    pico_vec_init(&vector, 25);
    pico_vec_init(&visited, 25);
    postorder(entry, &vector, &visited);

    // post-order gives [leaves ... entry]; reverse -> [entry ... leaves]
    pico_vec_reverse(&vector);

    // seed the entry node with grad 1
    struct PicoTensor* curr = NULL;
    curr = (struct PicoTensor*)vector.data[0];
    for(int i = 0; i < curr->numel; i++) {
        curr->grad[i] = 1.0f;
    }

    // call backward on each  (now iterate FORWARD: entry is first)
    for(int i = 0; i < vector.size; i++) {
        curr = (struct PicoTensor*)vector.data[i];
        if(curr->_backward != NULL) {
            curr->_backward(curr);
        }
    }

    pico_vec_free(&vector);
    pico_vec_free(&visited);
}

struct PicoTensor* pico_param(int64_t* shape, uint8_t ndim) {
    struct PicoTensor* tensor = (struct PicoTensor*)calloc(1, sizeof(struct PicoTensor));
    if(tensor == NULL) {
        printf("Memory allocation failed!\n");
        return NULL;
    }

    tensor->ndim = ndim;
    tensor->is_persistent = 1;

    // allocate and copy the shape array
    tensor->shape = (int64_t*)calloc(ndim, sizeof(int64_t));
    if(tensor->shape == NULL) {
        free(tensor);
        return NULL;
    }
    memcpy(tensor->shape, shape, ndim * sizeof(int64_t));

    // compute number of elements
    int numel = pico_compute_numel(tensor->shape, tensor->ndim);

    tensor->data = (float*)calloc(numel, sizeof(float));
    tensor->grad = (float*)calloc(numel, sizeof(float));
    tensor->strides = (int64_t*)calloc(tensor->ndim, sizeof(int64_t));

    // check if any inner allocations failed
    if(tensor->data == NULL || tensor->grad == NULL || tensor->strides == NULL) {
        free(tensor->shape);
        free(tensor->data);
        free(tensor->grad);
        free(tensor->strides);
        free(tensor);
        return NULL;
    }

    tensor->numel = numel;
    // compute strides using the freshly allocated array
    pico_compute_strides(shape, ndim, tensor->strides);

    return tensor;
}

struct PicoTensor* pico_create_tensor(struct Arena* arena, int64_t* shape, uint8_t ndim) {
    struct PicoTensor* tensor = (struct PicoTensor*)arena_alloc(arena, sizeof(struct PicoTensor));
    if(tensor == NULL) {
        printf("Memory allocation failed!\n");
        return NULL;
    }

    tensor->ndim = ndim;
    tensor->is_persistent = 0;

    // arena_alloc returns GARBAGE (not zeroed like calloc), so init these by hand
    // or the op/autograd code will read junk pointers.
    tensor->_backward = NULL;
    tensor->parents = NULL;
    tensor->num_parents = 0;
    tensor->backend = CPU;  // ops override this to inherit from inputs

    // allocate and copy the shape array
    tensor->shape = (int64_t*)arena_alloc(arena, (ndim * sizeof(int64_t)));
    if(tensor->shape == NULL) {
        free(tensor);
        return NULL;
    }
    memcpy(tensor->shape, shape, ndim * sizeof(int64_t));

    // compute number of elements
    int numel = pico_compute_numel(tensor->shape, tensor->ndim);

    tensor->data = (float*)arena_alloc(arena, numel * sizeof(float));
    memset(tensor->data, 0, numel * sizeof(float));
    tensor->grad = (float*)arena_alloc(arena, numel * sizeof(float));
    memset(tensor->grad, 0, numel * sizeof(float));
    tensor->strides = (int64_t*)arena_alloc(arena, tensor->ndim * sizeof(int64_t));

    // check if any inner allocations failed
    if(tensor->data == NULL || tensor->grad == NULL || tensor->strides == NULL) {
        free(tensor->shape);
        free(tensor->data);
        free(tensor->grad);
        free(tensor->strides);
        free(tensor);
        return NULL;
    }

    tensor->numel = numel;
    pico_compute_strides(shape, ndim, tensor->strides);

    return tensor;
}

void pico_free(struct PicoTensor* tensor) {
    // if the pointer is already NULL, do nothing safely
    if(tensor == NULL) {
        return;
    }

    // check if memory is in an arena
    if(tensor->is_persistent == 0) {
        return;
    }

    // free internal arrays first
    if(tensor->shape != NULL) {
        free(tensor->shape);
    }
    if(tensor->strides != NULL) {
        free(tensor->strides);
    }
    if(tensor->data != NULL) {
        free(tensor->data);
    }
    if(tensor->grad != NULL) {
        free(tensor->grad);
    }

    // free the dynamic parent array if it was allocated
    if(tensor->parents != NULL) {
        free(tensor->parents);
    }

    // free the main tensor structure
    free(tensor);
}

// a 1-element tensor holding `value`. shape {1} -> broadcasts against anything via
// map_index (the size-1 dim is stretched). leaf tensor: no parents, _backward NULL
// (pico_create_tensor already sets those), so it acts as a constant in the graph.
struct PicoTensor* pico_tensor_from_scalar(float value) {
    struct Arena* arena = arena_ctx_current();
    if(arena == NULL) {
        fprintf(stderr, "[Pico] Error: No current arena in context!\n");
        return NULL;
    }

    int64_t shape[1] = {1};
    struct PicoTensor* tensor = pico_create_tensor(arena, shape, 1);
    if(tensor == NULL) {
        return NULL;
    }
    tensor->data[0] = value;

    return tensor;
}

void pico_transpose_2d(struct PicoTensor* tensor) {
    if(tensor->ndim != 2) {
        fprintf(stderr, "Error: This is not a rank 2 tensor!\n");
        return;
    }

    // swap the r and c
    int c = tensor->shape[1];
    tensor->shape[1] = tensor->shape[0];
    tensor->shape[0] = c;

    int sc = tensor->strides[1];
    tensor->strides[1] = tensor->strides[0];
    tensor->strides[0] = sc;
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
void generate_random_floats_fast(float* arr, size_t size) {
    for(size_t i = 0; i < size; i++) {
        uint32_t r = xorshift32();
        // Construct a float in the range [1.0, 2.0) by setting mantissa bits
        uint32_t bits = (r >> 9) | 0x3F800000;
        float f = *(float*)&bits;
        arr[i] = f - 1.0f;  // Shift range down to [0.0, 1.0)
    }
}

struct PicoTensor* pico_rand(struct Arena* arena, int64_t* shape, uint8_t ndim) {
    struct PicoTensor* tensor = pico_create_tensor(arena, shape, ndim);
    // dispatch properly into backends
    generate_random_floats_fast(tensor->data, tensor->numel);
    return tensor;
}

// ============================= pico_randn

/*
 *dubs
 *import torch

# 1. PyTorch allocates space and generates flat uniform numbers [0, 1)
# For a requested size of 1000, it creates two blocks of 500
u1 = torch.rand(500)
u2 = torch.rand(500)

# 2. It applies the math to the entire tensor at once (Vectorization)
# No loops! The computer processes these arrays in parallel rows.
mag = torch.sqrt(-2.0 * torch.log(u1))
angle = 2.0 * torch.pi * u2

# 3. Box-Muller gives TWO normal numbers per pair using cos and sin
z0 = mag * torch.cos(angle)
z1 = mag * torch.sin(angle)

# 4. Combine them into the final 1000-element tensor you asked for
final_randn_tensor = torch.cat([z0, z1])

 *
 * */

struct PicoTensor* pico_randn(struct Arena* arena, int64_t* shape, uint8_t ndim) {
    int64_t* res_shape = arena_alloc(arena, sizeof(int64_t) * ndim);
    memcpy(res_shape, shape, sizeof(int64_t) * ndim);
    res_shape[ndim - 1] = res_shape[ndim - 1] / 2;

    struct PicoTensor* u1 = pico_rand(arena, res_shape, ndim);
    struct PicoTensor* u2 = pico_rand(arena, res_shape, ndim);

    struct PicoTensor* mag =
        pico_tensor_sqrt(pico_mul(pico_tensor_from_scalar(-2.0), pico_tensor_log(u1)));
    struct PicoTensor* angle =
        pico_mul(pico_tensor_from_scalar(2.0), pico_mul(pico_tensor_from_scalar(PI_F), u2));

    struct PicoTensor* z0 = pico_mul(mag, pico_tensor_cos(angle));
    struct PicoTensor* z1 = pico_mul(mag, pico_tensor_sin(angle));

    struct PicoTensor* tensor = pico_cat(z0, z1, 0);

    return tensor;
}

// ============================= end

uint8_t pico_check_broadcast_compatibility(struct PicoTensor* a, struct PicoTensor* b) {
    int ndim_a = a->ndim;
    int ndim_b = b->ndim;

    // We check from the end of the shape arrays (the "trailing" dimensions)
    int i = ndim_a - 1;
    int j = ndim_b - 1;

    while(i >= 0 && j >= 0) {
        int dim_a = a->shape[i];
        int dim_b = b->shape[j];

        // The Broadcasting Rule:
        // 1. Dimensions are equal, OR
        // 2. One of them is 1
        if(dim_a != dim_b && dim_a != 1 && dim_b != 1) {
            return 0;  // Not compatible!
        }
        i--;
        j--;
    }

    // If one tensor has more dimensions (e.g., [5, 4, 3] vs [4, 3]),
    // the extra leading dimensions [5] are always compatible with
    // the "implicit ones" of the smaller tensor.
    return 1;
}

void postorder(struct PicoTensor* root, struct PicoVec* vector, struct PicoVec* visited) {
    if(root == NULL) {
        return;
    }
    if(pico_vec_find(visited, root) != -1) {  // if node was found? stop redundant traversals
        return;
    }

    pico_vec_push(visited, root);

    for(int i = 0; i < root->num_parents; i++) {
        postorder(root->parents[i], vector, visited);
    }

    // append to array if not appended before
    pico_vec_push(vector, root);
}
