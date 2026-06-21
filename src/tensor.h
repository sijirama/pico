#pragma once

#include <stdint.h>

struct PicoTensor {  // 51 + 5b padding = 56b
    int64_t* shape;
    int64_t* strides;
    float* data;
    float* grad;
    void (*_backward)(struct PicoTensor*);
    struct PicoTensor** parents;
    uint8_t ndim;
    uint8_t num_parents;
    uint8_t is_persistent;  // memory malloc'd ?
};

struct PicoTensor* pico_param(int64_t* shape, uint8_t ndim);
void pico_free(struct PicoTensor* tensor);

// ============================= helpers

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
