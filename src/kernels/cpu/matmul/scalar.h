#pragma once

#include "../../../tensor.h"

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

static inline void pico_grouped_matmul_cpu_scalar(struct PicoTensor* a, struct PicoTensor* b, struct PicoTensor* out,
                                                  int group_size) {
    int batch_count = a->shape[0];
    int query_heads = a->shape[1];
    int rows = a->shape[2];
    int k_dim = a->shape[3];
    int columns = b->shape[3];

    for(int batch = 0; batch < batch_count; batch++) {
        for(int q_head = 0; q_head < query_heads; q_head++) {
            int kv_head = q_head / group_size;
            for(int i = 0; i < rows; i++) {
                for(int k = 0; k < k_dim; k++) {
                    float m_cell = a->data[batch * a->strides[0] + q_head * a->strides[1] +
                                           i * a->strides[2] + k * a->strides[3]];
                    for(int j = 0; j < columns; j++) {
                        out->data[batch * out->strides[0] + q_head * out->strides[1] +
                                  i * out->strides[2] + j * out->strides[3]] +=
                            m_cell * b->data[batch * b->strides[0] + kv_head * b->strides[1] +
                                             k * b->strides[2] + j * b->strides[3]];
                    }
                }
            }
        }
    }
}
