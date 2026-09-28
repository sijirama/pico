#pragma once

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "../arena.h"
#include "../ctx.h"
#include "../ops.h"
#include "../tensor.h"
#include "../tensor_ops.h"

struct PicoContext;

struct PicoAttn {
    struct PicoTensor *Q; // [embed_dim, num_heads * d_k]
    struct PicoTensor *K; // [embed_dim, num_heads * d_k]
    struct PicoTensor *V; // [embed_dim, num_heads * d_k]
    struct PicoTensor *O; // [num_heads * d_k, embed_dim]
    int embed_dim;
    int num_of_heads;
    int d_k;
    int swa_window;
    int gqa_group_size;
};

struct PicoAttn *pico_nn_attn_init(struct PicoContext *ctx, char *name, int embed_dim, int num_heads, int d_k);
struct PicoAttn *pico_nn_swa_attn_init(struct PicoContext *ctx, char *name, int embed_dim, int num_heads, int d_k, int window);
struct PicoAttn *pico_nn_gqa_attn_init(struct PicoContext *ctx, char *name, int embed_dim, int num_heads, int d_k, int group_size);

struct PicoTensor *pico_nn_attn_forward(struct PicoContext *ctx, struct PicoAttn *attn, struct PicoTensor *input);
struct PicoTensor *pico_nn_swa_attn_forward(struct PicoContext *ctx, struct PicoAttn *attn, struct PicoTensor *input);
struct PicoTensor *pico_nn_gqa_attn_forward(struct PicoContext *ctx, struct PicoAttn *attn, struct PicoTensor *input);

// utils ================================================================

static int64_t pico_attn_rope_offset(struct PicoTensor *tensor, int64_t batch, int64_t pos, int64_t head, int64_t head_i) {
    if(tensor->ndim == 2) {
        return pos * tensor->strides[0] + (head + head_i) * tensor->strides[1];
    }
    if(tensor->ndim == 3) {
        return pos * tensor->strides[0] + head * tensor->strides[1] + head_i * tensor->strides[2];
    }

    return batch * tensor->strides[0] + pos * tensor->strides[1] + head * tensor->strides[2] + head_i * tensor->strides[3];
}

static inline void pico_nn_attn_apply_rope(struct PicoTensor *tensor, int num_heads, int d_k) {
    if(tensor == NULL || num_heads <= 0 || d_k <= 0) {
        return;
    }

    if((d_k % 2) != 0) {
        fprintf(stderr, "PicoAttentionError: rope needs an even d_k\n");
        return;
    }

    int64_t batch_count = 1;
    int64_t seq_len = 0;
    int64_t head_stride = 0;

    if(tensor->ndim == 2) {
        if(tensor->shape[1] != num_heads * d_k) {
            fprintf(stderr, "PicoAttentionError: rope expected [seq, num_heads * d_k]\n");
            return;
        }

        seq_len = tensor->shape[0];
        head_stride = d_k;
    } else if(tensor->ndim == 3) {
        if(tensor->shape[1] != num_heads || tensor->shape[2] != d_k) {
            fprintf(stderr, "PicoAttentionError: rope expected [seq, num_heads, d_k]\n");
            return;
        }

        seq_len = tensor->shape[0];
        head_stride = 1;
    } else if(tensor->ndim == 4) {
        if(tensor->shape[2] != num_heads || tensor->shape[3] != d_k) {
            fprintf(
                stderr,
                "PicoAttentionError: rope expected [batch, seq, num_heads, "
                "d_k]\n");
            return;
        }

        batch_count = tensor->shape[0];
        seq_len = tensor->shape[1];
        head_stride = 1;
    } else {
        fprintf(stderr, "PicoAttentionError: rope only supports 2d, 3d, or 4d tensors\n");
        return;
    }

    for(int64_t b = 0; b < batch_count; b++) {
        for(int64_t pos = 0; pos < seq_len; pos++) {
            for(int64_t h = 0; h < num_heads; h++) {
                int64_t head_base = h * head_stride;

                for(int64_t i = 0; i < d_k; i += 2) {
                    float inv_freq = 1.0f / powf(10000.0f, (float)i / (float)d_k);
                    float angle = (float)pos * inv_freq;
                    float cos_v = cosf(angle);
                    float sin_v = sinf(angle);

                    int64_t even_offset = pico_attn_rope_offset(tensor, b, pos, head_base, i);
                    int64_t odd_offset = pico_attn_rope_offset(tensor, b, pos, head_base, i + 1);

                    float even = tensor->data[even_offset];
                    float odd = tensor->data[odd_offset];

                    tensor->data[even_offset] = even * cos_v - odd * sin_v;
                    tensor->data[odd_offset] = even * sin_v + odd * cos_v;
                }
            }
        }
    }
}

static inline char *pico_attn_param_name(struct Arena *arena, char *name, char *suffix) {
    size_t len = strlen(name) + strlen(suffix) + 1;
    char *param_name = (char *)arena_alloc(arena, len);
    if(param_name == NULL) {
        return NULL;
    }

    strcpy(param_name, name);
    strcat(param_name, suffix);
    return param_name;
}

static inline void pico_nn_attn_free(struct PicoAttn *attn) {
    if(attn == NULL) {
        return;
    }

    free(attn);
}

static inline void pico_nn_attn_causal_mask(struct PicoTensor *table) {
    if(table == NULL || table->data == NULL) {
        return;
    }

    // Expected attention score shape:
    // (num_heads, S, S) or (B, num_heads, S, S)
    if(table->ndim != 3 && table->ndim != 4) {
        fprintf(stderr, "PicoAttentionError: causal mask expects a 3D or 4D tensor\n");
        return;
    }

    int64_t B = table->ndim == 4 ? table->shape[0] : 1;
    int64_t H = table->ndim == 4 ? table->shape[1] : table->shape[0];
    int64_t Q = table->ndim == 4 ? table->shape[2] : table->shape[1];
    int64_t K = table->ndim == 4 ? table->shape[3] : table->shape[2];

    // For ordinary causal self-attention, Q == K == S.
    if(Q != K) {
        fprintf(
            stderr,
            "PicoAttentionError: causal mask expects square attention "
            "scores\n");
        return;
    }

    for(int64_t b = 0; b < B; b++) {
        for(int64_t h = 0; h < H; h++) {
            for(int64_t q = 0; q < Q; q++) {
                for(int64_t k = q + 1; k < K; k++) {
                    int64_t offset = table->ndim == 4
                                         ? b * table->strides[0] + h * table->strides[1] + q * table->strides[2] + k * table->strides[3]
                                         : h * table->strides[0] + q * table->strides[1] + k * table->strides[2];

                    table->data[offset] = -INFINITY;
                }
            }
        }
    }
}

static inline void pico_nn_attn_swa_causal_mask(struct PicoTensor *table, int window) {
    if(table == NULL || table->data == NULL) {
        return;
    }

    // Expected attention score shape:
    // (num_heads, S, S) or (B, num_heads, S, S)
    if(table->ndim != 3 && table->ndim != 4) {
        fprintf(stderr, "PicoAttentionError: causal mask expects a 3D or 4D tensor\n");
        return;
    }

    int64_t B = table->ndim == 4 ? table->shape[0] : 1;
    int64_t H = table->ndim == 4 ? table->shape[1] : table->shape[0];
    int64_t Q = table->ndim == 4 ? table->shape[2] : table->shape[1];
    int64_t K = table->ndim == 4 ? table->shape[3] : table->shape[2];

    // For ordinary causal self-attention, Q == K == S.
    if(Q != K) {
        fprintf(stderr, "PicoAttentionError: causal mask expects square attention scores\n");
        return;
    }

    for(int64_t b = 0; b < B; b++) {
        for(int64_t h = 0; h < H; h++) {
            for(int64_t q = 0; q < Q; q++) {
                for(int64_t k = 0; k < K; k++) {
                    int64_t offset = table->ndim == 4
                                         ? b * table->strides[0] + h * table->strides[1] + q * table->strides[2] + k * table->strides[3]
                                         : h * table->strides[0] + q * table->strides[1] + k * table->strides[2];

                    if(k > q || k < q - window) {
                        table->data[offset] = -INFINITY;
                    }
                }
            }
        }
    }
}
