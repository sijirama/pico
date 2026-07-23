#pragma once

#include <stdint.h>
#include <string.h>
#include <stdbool.h>
#include "arena.h"

#define PI_F 3.14159265358979323846f  // M_PI isn't exposed under -std=c11
typedef enum { CPU, GPU } PicoBackend;

struct PicoTensor {
    int64_t* shape;
    int64_t* strides;
    float* data;
    float* grad;
    void (*_backward)(struct PicoTensor*);
    struct PicoTensor** parents;
    int64_t numel;
    PicoBackend backend;
    uint8_t ndim;
    uint8_t num_parents;
    uint8_t is_persistent;  // memory malloc'd ?
};

void pico_backward(struct Arena* arena, struct PicoTensor* entry);

struct PicoTensor* pico_param(int64_t* shape, uint8_t ndim);
struct PicoTensor* pico_create_tensor(struct Arena* arena, int64_t* shape, uint8_t ndim);

// a 1-element tensor (shape {1}) holding a single scalar. broadcasts against any
// shape, so you can do pico_mul(pico_tensor_from_scalar(2.0f), t). uses the current
// ctx arena (composes with the ops, which also use arena_ctx_current()).
struct PicoTensor* pico_tensor_from_scalar(float value);

void pico_free(struct PicoTensor* tensor);

// pretty-print a tensor's shape + data, nested by shape (respects strides).
void pico_tensor_print(struct PicoTensor* t);

// ====================================== important ops
// TODO: siji don't forget about these guys

void pico_transpose(struct PicoTensor* tensor);
void pico_transpose_2d(struct PicoTensor* tensor);

void pico_transpose_reshape(struct PicoTensor* tensor, int64_t* shape, int ndim);

struct PicoTensor* pico_transpose_clone(struct PicoTensor* tensor);

struct PicoTensor* pico_cat(struct PicoTensor* a, struct PicoTensor* b, int dim);

// returns a tensor filled with random numbers from a uniform distribution on the interval ([0,1])
struct PicoTensor* pico_rand(struct Arena* arena, int64_t* shape, uint8_t ndim);

// returns a tensor filled with random numbers drawn from a standard normal distribution (Gaussian
// distribution) with a mean of 0 and a variance (and standard deviation) of 1
struct PicoTensor* pico_randn(struct Arena* arena, int64_t* shape, uint8_t ndim);

// ============================= helpers

#define MAX(a, b) (((a) > (b)) ? (a) : (b))
#define MIN(a, b) (((a) < (b)) ? (a) : (b))

static inline int pico_compute_numel(int64_t* shape, int ndim) {
    int numel = 1;
    for(int i = 0; i < ndim; i++) {
        numel *= shape[i];
    }
    return numel;
}

static inline void pico_compute_strides(int64_t* shape, int ndim, int64_t* strides_out) {
    strides_out[ndim - 1] = 1;
    for(int i = ndim - 2; i >= 0; i--) {
        strides_out[i] = strides_out[i + 1] * shape[i + 1];
    }
}

static inline void pico_tensor_update_strides(struct PicoTensor* t) {
    int current_stride = 1;
    for(int i = t->ndim - 1; i >= 0; i--) {
        t->strides[i] = current_stride;
        current_stride *= t->shape[i];
    }
}

static inline bool pico_tensor_shapes_are_equal(struct PicoTensor* a, struct PicoTensor* b) {
    if (a->ndim != b->ndim) {
        return false;
    }
    return memcmp(a->shape, b->shape, sizeof(int64_t) * a->ndim) == 0;
}

// ============================= broadcasting

uint8_t pico_check_broadcast_compatibility(struct PicoTensor* a, struct PicoTensor* b);

// pad a tensor's shape up to `ndim` by prepending 1s on the left (right-align).
// only used in ops to compute the output shape (out_dim = max of padded shapes).
static inline int64_t* pad_shape(struct Arena* arena, struct PicoTensor* smaller, int ndim) {
    int64_t* padded = arena_alloc(arena, sizeof(int64_t) * ndim);
    int diff = ndim - smaller->ndim;
    for(int i = 0; i < ndim; i++) {
        if(i < diff) {
            padded[i] = 1;
        } else {
            // subtract 'diff' to map the large index back to the small one
            padded[i] = smaller->shape[i - diff];
        }
    }
    return padded;
}

// given an output element index `global_i`, return the flat index into tensor `t`
// it should read from, handling broadcasting directly (no padded arrays needed).
//   - unravel global_i into coords using the (contiguous) output strides
//   - for each output dim, find the matching source dim (offset by `diff`)
//   - a prepended dim (sd < 0) or a size-1 dim contributes 0 -> reuse (the stretch)
//
//   give us an index of t that's the same with global_index when the stride has been stretched to
//   match out - sijirama
static inline int64_t map_index(int64_t global_i, struct PicoTensor* t, int64_t* out_strides,
                                int out_ndim) {
    int64_t mapped_idx = 0;
    int64_t rem = global_i;
    int diff = out_ndim - t->ndim;
    for(int d = 0; d < out_ndim; d++) {
        int64_t coord = rem / out_strides[d];
        rem %= out_strides[d];

        int sd = d - diff;  // matching dim in the source tensor
        if(sd >= 0 && t->shape[sd] > 1) {
            mapped_idx += coord * t->strides[sd];
        }
        // else: prepended dim or size-1 dim -> contributes nothing -> reuse element
    }
    return mapped_idx;
}
