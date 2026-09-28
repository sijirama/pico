#include "attn.h"

struct PicoAttn *pico_nn_gqa_attn_init(struct PicoContext *ctx, char *name, int embed_dim, int num_heads, int d_k, int group_size) {

    if(ctx == NULL || name == NULL || embed_dim <= 0 || num_heads <= 0 || d_k <= 0) {
        return NULL;
    }

    struct Arena *arena = pico_context_arena(ctx);
    if(arena == NULL) {
        fprintf(
            stderr,
            "PicoArenaError: no arena available for attention init "
            "allocation\n");
        return NULL;
    }

    struct PicoAttn *attn = malloc(sizeof(struct PicoAttn));
    if(attn == NULL) {
        perror("Failed to allocate PicoAttn");
        return NULL;
    }

    attn->gqa_group_size = group_size;
    attn->embed_dim = embed_dim;
    attn->num_of_heads = num_heads;
    attn->d_k = d_k;

    int num_kv_heads = num_heads / group_size;

    int64_t q_shape[2] = {embed_dim, num_heads * d_k};
    int64_t k_shape[2] = {embed_dim, num_kv_heads * d_k};
    int64_t v_shape[2] = {embed_dim, num_kv_heads * d_k};
    int64_t o_shape[2] = {num_heads * d_k, embed_dim};

    char *q_name = pico_attn_param_name(arena, name, ".q_proj.weight");
    char *k_name = pico_attn_param_name(arena, name, ".k_proj.weight");
    char *v_name = pico_attn_param_name(arena, name, ".v_proj.weight");
    char *o_name = pico_attn_param_name(arena, name, ".out_proj.weight");

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

struct PicoTensor *pico_nn_gqa_attn_forward(struct PicoContext *ctx, struct PicoAttn *attn, struct PicoTensor *input) {
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

    int num_kv_heads = attn->num_of_heads / attn->gqa_group_size;

    if(input->ndim == 2) {
        res_shape[0] = K->shape[0];
        res_shape[1] = num_kv_heads;
        res_shape[2] = attn->d_k;
    } else {
        res_shape[0] = K->shape[0];
        res_shape[1] = K->shape[1];
        res_shape[2] = num_kv_heads;
        res_shape[3] = attn->d_k;
    }

    pico_view(ctx, K, res_shape, ndim); // (B, S, num_kv_heads, d_k)
    pico_view(ctx, V, res_shape, ndim); // (B, S, num_kv_heads, d_k)

    pico_nn_attn_apply_rope(Q, attn->num_of_heads, attn->d_k);
    pico_nn_attn_apply_rope(K, num_kv_heads, attn->d_k);

    int64_t permute_dims[4] = {0, 2, 1, 3};
    int64_t permute_dims_2d[3] = {1, 0, 2};
    int64_t *qkv_permute = input->ndim == 2 ? permute_dims_2d : permute_dims;
    pico_permute(ctx, Q, qkv_permute); // (B, num_heads, S, d_k)
    pico_permute(ctx, K, qkv_permute); // (B, num_kv_heads, S, d_k)
    pico_permute(ctx, V, qkv_permute); // (B, num_kv_heads, S, d_k)

    // transpose with permute
    int64_t k_transpose_dims[4] = {0, 1, 3, 2};
    int64_t k_transpose_dims_2d[3] = {0, 2, 1};
    int64_t *k_permute = input->ndim == 2 ? k_transpose_dims_2d : k_transpose_dims;
    pico_permute(ctx, K, k_permute); // (B, num_kv_heads, d_k, S)

    // Q:    (B, num_heads, S, d_k)
    // K^T:  (B, num_kv_heads, d_k, S)
    //
    // QK_t: (B, num_heads, S, S)
    struct PicoTensor *QK_t = pico_grouped_matmul(ctx, Q, K, attn->gqa_group_size);

    struct PicoTensor *d_k_saclar = pico_tensor_from_scalar(ctx, (1 / sqrtf(attn->d_k)));
    struct PicoTensor *QK_scaled = pico_mul(ctx, QK_t, d_k_saclar); // (B, num_heads, S, S)
    pico_nn_attn_causal_mask(QK_scaled);
    struct PicoTensor *A = pico_softmax(ctx, QK_scaled, input->ndim == 2 ? 2 : 3);

    struct PicoTensor *O = pico_grouped_matmul(ctx, A, V, attn->gqa_group_size); // (B, num_heads, S, d_k)

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
