#include "attn.h"

struct PicoAttn *pico_nn_swa_attn_init(struct PicoContext *ctx, char *name, int embed_dim, int num_heads, int d_k, int window) {
    struct PicoAttn *attn = pico_nn_attn_init(ctx, name, embed_dim, num_heads, d_k);
    attn->swa_window = window;
    return attn;
}

struct PicoTensor *pico_nn_swa_attn_forward(struct PicoContext *ctx, struct PicoAttn *attn, struct PicoTensor *input) {
    struct PicoTensor *Q = pico_matmul(ctx, input, attn->Q); // input (B x S x E) @ Q_w (E x d_k)
    struct PicoTensor *K = pico_matmul(ctx, input, attn->K); // input (B x S x E) @ K_w (E x d_k)
    struct PicoTensor *V = pico_matmul(ctx, input, attn->V); // input (B x S x E) @ V_w (E x d_k)

    if(Q == NULL || K == NULL || V == NULL) {
        return NULL;
    }

    if(input->ndim != 2 && input->ndim != 3) {
        fprintf(stderr, "PicoAttentionError: forward only supports 2D or 3D input\n");
        return NULL;
    }

    int ndim = input->ndim + 1;
    int64_t *res_shape = arena_alloc(ctx->arena, sizeof(int64_t) * 4);
    if(res_shape == NULL) {
        return NULL;
    }

    if(input->ndim == 2) {
        res_shape[0] = Q->shape[0];
        res_shape[1] = attn->num_of_heads;
        res_shape[2] = attn->d_k;
    } else {
        res_shape[0] = Q->shape[0];
        res_shape[1] = Q->shape[1];
        res_shape[2] = attn->num_of_heads;
        res_shape[3] = attn->d_k;
    }

    pico_view(ctx, Q, res_shape, ndim); // (B, S, num_heads, d_k)
    pico_view(ctx, K, res_shape, ndim); // (B, S, num_heads, d_k)
    pico_view(ctx, V, res_shape, ndim); // (B, S, num_heads, d_k)

    pico_nn_attn_apply_rope(Q, attn->num_of_heads, attn->d_k);
    pico_nn_attn_apply_rope(K, attn->num_of_heads, attn->d_k);

    int64_t permute_dims[4] = {0, 2, 1, 3};
    int64_t permute_dims_2d[3] = {1, 0, 2};
    int64_t *qkv_permute = input->ndim == 2 ? permute_dims_2d : permute_dims;
    pico_permute(ctx, Q, qkv_permute); // (B, num_heads, S, d_k)
    pico_permute(ctx, K, qkv_permute); // (B, num_heads, S, d_k)
    pico_permute(ctx, V, qkv_permute); // (B, num_heads, S, d_k)

    // transpose with permute
    int64_t k_transpose_dims[4] = {0, 1, 3, 2};
    int64_t k_transpose_dims_2d[3] = {0, 2, 1};
    int64_t *k_permute = input->ndim == 2 ? k_transpose_dims_2d : k_transpose_dims;
    pico_permute(ctx, K, k_permute); // (B, num_heads, d_k, S)

    // Q:    (B, num_heads, S, d_k)
    // K^T:  (B, num_heads, d_k, S)
    //
    // QK_t: (B, num_heads, S, S)
    struct PicoTensor *QK_t = pico_swa_matmul(ctx, Q, K, attn->swa_window);

    struct PicoTensor *d_k_saclar = pico_tensor_from_scalar(ctx, (1 / sqrtf(attn->d_k)));
    struct PicoTensor *QK_scaled = pico_mul(ctx, QK_t, d_k_saclar); // (B, num_heads, S, S)

    struct PicoTensor *A = pico_causal_softmax(ctx, QK_scaled, input->ndim == 2 ? 2 : 3, attn->swa_window);

    struct PicoTensor *O = pico_matmul(ctx, A, V); // (B, num_heads, S, d_k)

    int64_t permute_dims_back[4] = {0, 2, 1, 3};
    int64_t permute_dims_back_2d[3] = {1, 0, 2};
    int64_t *o_permute = input->ndim == 2 ? permute_dims_back_2d : permute_dims_back;
    pico_permute(ctx, O, o_permute); // (B, S, num_heads, d_k)

    ndim = input->ndim;
    if(input->ndim == 2) {
        res_shape[0] = O->shape[0];
        res_shape[1] = attn->num_of_heads * attn->d_k;
    } else {
        res_shape[0] = O->shape[0];
        res_shape[1] = O->shape[1];
        res_shape[2] = attn->num_of_heads * attn->d_k;
    }
    pico_view(ctx, O, res_shape, ndim); // (B, S, num_heads * d_k) = (B,S,embed_dim)

    struct PicoTensor *final = pico_matmul(ctx, O, attn->O); // (B,S,embed_dim)

    return final;
}
