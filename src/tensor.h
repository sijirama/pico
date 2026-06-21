#pragma once

#include <stdint.h>

struct PicoTensor { // 50 + 6b padding = 56b
    int64_t* shape;
    int64_t* strides;
    float* data;
    float* grad;
    void (*_backward)(struct PicoTensor*);
    struct PicoTensor** parents;
    uint8_t ndim;
    uint8_t num_parents;
};
