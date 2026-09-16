#include "self-attn.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "arena.h"
#include "ops.h"
#include "tensor.h"
#include "tensor_ops.h"

void pico_nn_attn_causal_mask(struct PicoTensor* table) {
    if(table == NULL || table->data == NULL) {
        return;
    }

    // Expected attention score shape:
    // (B, num_heads, S, S)
    if(table->ndim != 4) {
        fprintf(stderr, "PicoAttentionError: causal mask expects a 4D tensor\n");
        return;
    }

    int64_t B = table->shape[0];
    int64_t H = table->shape[1];
    int64_t Q = table->shape[2];
    int64_t K = table->shape[3];

    // For ordinary causal self-attention, Q == K == S.
    if(Q != K) {
        fprintf(stderr, "PicoAttentionError: causal mask expects square attention scores\n");
        return;
    }

    for(int64_t b = 0; b < B; b++) {
        for(int64_t h = 0; h < H; h++) {
            for(int64_t q = 0; q < Q; q++) {
                for(int64_t k = q + 1; k < K; k++) {
                    int64_t offset =
                        b * table->strides[0] + h * table->strides[1] + q * table->strides[2] + k * table->strides[3];

                    table->data[offset] = -INFINITY;
                }
            }
        }
    }
}

static int64_t pico_attn_rope_offset(struct PicoTensor* tensor, int64_t batch, int64_t pos, int64_t head,
                                     int64_t head_i) {
    if(tensor->ndim == 2) {
        return pos * tensor->strides[0] + (head + head_i) * tensor->strides[1];
    }
    if(tensor->ndim == 3) {
        return pos * tensor->strides[0] + head * tensor->strides[1] + head_i * tensor->strides[2];
    }

    return batch * tensor->strides[0] + pos * tensor->strides[1] + head * tensor->strides[2] +
           head_i * tensor->strides[3];
}

void pico_nn_attn_apply_rope(struct PicoTensor* tensor, int num_heads, int d_k) {
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
            fprintf(stderr, "PicoAttentionError: rope expected [batch, seq, num_heads, d_k]\n");
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

struct PicoTensor* pico_nn_attn_forward(struct PicoContext* ctx, struct PicoAttn* attn, struct PicoTensor* input) {
    struct PicoTensor* Q = pico_matmul(ctx, input, attn->Q);  // input (B x S x E) @ Q_w (E x d_k)
    struct PicoTensor* K = pico_matmul(ctx, input, attn->K);  // input (B x S x E) @ K_w (E x d_k)
    struct PicoTensor* V = pico_matmul(ctx, input, attn->V);  // input (B x S x E) @ V_w (E x d_k)

    pico_nn_attn_apply_rope(Q, attn->num_of_heads, attn->d_k);
    pico_nn_attn_apply_rope(K, attn->num_of_heads, attn->d_k);

    int ndim = 4;
    int64_t* res_shape = arena_alloc(ctx->arena, sizeof(int64_t) * ndim);
    res_shape[0] = Q->shape[0];
    res_shape[1] = Q->shape[1];
    res_shape[2] = attn->num_of_heads;
    res_shape[3] = attn->d_k;

    pico_view(ctx, Q, res_shape, ndim);  // (B, S, num_heads, d_k)
    pico_view(ctx, K, res_shape, ndim);  // (B, S, num_heads, d_k)
    pico_view(ctx, V, res_shape, ndim);  // (B, S, num_heads, d_k)

    int64_t permute_dims[4] = {0, 2, 1, 3};
    pico_permute(ctx, Q, permute_dims);  // (B, num_heads, S, d_k)
    pico_permute(ctx, K, permute_dims);  // (B, num_heads, S, d_k)
    pico_permute(ctx, V, permute_dims);  // (B, num_heads, S, d_k)

    // transpose with permute
    int64_t k_transpose_dims[4] = {0, 1, 3, 2};
    pico_permute(ctx, K, k_transpose_dims);  // (B, num_heads, d_k, S)

    // Q:    (B, num_heads, S, d_k)
    // K^T:  (B, num_heads, d_k, S)
    //
    // QK_t: (B, num_heads, S, S)
    struct PicoTensor* QK_t = pico_matmul(ctx, Q, K);

    struct PicoTensor* d_k_saclar = pico_tensor_from_scalar(ctx, (1 / sqrtf(attn->d_k)));
    struct PicoTensor* QK_scaled = pico_mul(ctx, QK_t, d_k_saclar);  // (B, num_heads, S, S)
    pico_nn_attn_causal_mask(QK_scaled);
    struct PicoTensor* A = pico_softmax(ctx, QK_scaled, 1);

    struct PicoTensor* O = pico_matmul(ctx, A, V);  // (B, num_heads, S, d_k)

    int64_t permute_dims_back[4] = {0, 2, 1, 3};
    pico_permute(ctx, O, permute_dims_back);  // (B, S, num_heads, d_k)

    ndim = 3;
    res_shape[0] = O->shape[0];
    res_shape[1] = O->shape[1];
    res_shape[2] = attn->num_of_heads * attn->d_k;
    pico_view(ctx, O, res_shape, ndim);  // (B, S, num_heads * d_k) = (B,S,embed_dim)

    struct PicoTensor* final = pico_matmul(ctx, O, attn->O);  // (B,S,embed_dim)

    return final;
}

static char* pico_attn_param_name(struct Arena* arena, char* name, char* suffix) {
    size_t len = strlen(name) + strlen(suffix) + 1;
    char* param_name = arena_alloc(arena, len);
    if(param_name == NULL) {
        return NULL;
    }

    strcpy(param_name, name);
    strcat(param_name, suffix);
    return param_name;
}

struct PicoAttn* pico_nn_attn_init(struct PicoContext* ctx, char* name, int embed_dim, int num_heads, int d_k) {
    if(ctx == NULL || name == NULL || embed_dim <= 0 || num_heads <= 0 || d_k <= 0) {
        return NULL;
    }

    struct Arena* arena = pico_context_arena(ctx);
    if(arena == NULL) {
        fprintf(stderr, "PicoArenaError: no arena available for attention init allocation\n");
        return NULL;
    }

    struct PicoAttn* attn = malloc(sizeof(struct PicoAttn));
    if(attn == NULL) {
        perror("Failed to allocate PicoAttn");
        return NULL;
    }

    attn->embed_dim = embed_dim;
    attn->num_of_heads = num_heads;
    attn->d_k = d_k;

    int head_dim = num_heads * d_k;
    int64_t q_shape[2] = {embed_dim, head_dim};
    int64_t k_shape[2] = {embed_dim, head_dim};
    int64_t v_shape[2] = {embed_dim, head_dim};
    int64_t o_shape[2] = {head_dim, embed_dim};

    char* q_name = pico_attn_param_name(arena, name, ".q_proj.weight");
    char* k_name = pico_attn_param_name(arena, name, ".k_proj.weight");
    char* v_name = pico_attn_param_name(arena, name, ".v_proj.weight");
    char* o_name = pico_attn_param_name(arena, name, ".out_proj.weight");

    if(q_name == NULL || k_name == NULL || v_name == NULL || o_name == NULL) {
        free(attn);
        return NULL;
    }

    attn->Q = pico_param_named(ctx, q_name, q_shape, 2);
    attn->K = pico_param_named(ctx, k_name, k_shape, 2);
    attn->V = pico_param_named(ctx, v_name, v_shape, 2);
    attn->O = pico_param_named(ctx, o_name, o_shape, 2);

    if(attn->Q == NULL || attn->K == NULL || attn->V == NULL || attn->O == NULL) {
        free(attn);
        return NULL;
    }

    return attn;
}

void pico_nn_attn_free(struct PicoAttn* attn) {
    if(attn == NULL) {
        return;
    }

    free(attn);
}
