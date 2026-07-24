#pragma once

#include "tensor.h"

static inline void pico_matmul_cpu_scalar(struct PicoTensor* a, struct PicoTensor* b, struct PicoTensor* out) {
    int rows = a->shape[0];
    int columns = b->shape[1];
    int k_dim = a->shape[1];

    for(int i = 0; i < rows; i++) {
        for(int k = 0; k < k_dim; k++) {
            float m_cell = a->data[i * a->strides[0] + k * a->strides[1]];
            for(int j = 0; j < columns; j++) {
                out->data[i * out->strides[0] + j * out->strides[1]] +=
                    m_cell * b->data[k * b->strides[0] + j * b->strides[1]];
            }
        }
    }
}
