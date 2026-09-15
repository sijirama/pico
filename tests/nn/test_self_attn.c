#include "utest.h"

#include <math.h>

#include "global.h"
#include "nn/self-attn.h"
#include "tensor.h"

#define ASSERT_NEAR_FLOAT(actual, expected) ASSERT_NEAR((actual), (expected), 1e-5f)

UTEST(self_attn, init_sets_dims_and_projection_shapes) {
    struct PicoContext* ctx = pico_init_verbose(false);

    struct PicoAttn* attn = pico_nn_attn_init(ctx, "block.attn", 12, 3, 4);

    ASSERT_TRUE(attn != NULL);
    ASSERT_EQ(attn->embed_dim, 12);
    ASSERT_EQ(attn->num_of_heads, 3);
    ASSERT_EQ(attn->d_k, 4);

    ASSERT_EQ(attn->Q->shape[0], 12);
    ASSERT_EQ(attn->Q->shape[1], 12);
    ASSERT_EQ(attn->K->shape[0], 12);
    ASSERT_EQ(attn->K->shape[1], 12);
    ASSERT_EQ(attn->V->shape[0], 12);
    ASSERT_EQ(attn->V->shape[1], 12);
    ASSERT_EQ(attn->O->shape[0], 12);
    ASSERT_EQ(attn->O->shape[1], 12);

    pico_nn_attn_free(attn);
    pico_shutdown(ctx);
}

UTEST(self_attn, init_allows_head_projection_dim_different_from_embed_dim) {
    struct PicoContext* ctx = pico_init_verbose(false);

    struct PicoAttn* attn = pico_nn_attn_init(ctx, "wide.attn", 8, 4, 3);

    ASSERT_TRUE(attn != NULL);
    ASSERT_EQ(attn->Q->shape[0], 8);
    ASSERT_EQ(attn->Q->shape[1], 12);
    ASSERT_EQ(attn->K->shape[0], 8);
    ASSERT_EQ(attn->K->shape[1], 12);
    ASSERT_EQ(attn->V->shape[0], 8);
    ASSERT_EQ(attn->V->shape[1], 12);
    ASSERT_EQ(attn->O->shape[0], 12);
    ASSERT_EQ(attn->O->shape[1], 8);

    pico_nn_attn_free(attn);
    pico_shutdown(ctx);
}

UTEST(self_attn, init_registers_named_params) {
    struct PicoContext* ctx = pico_init_verbose(false);

    struct PicoAttn* attn = pico_nn_attn_init(ctx, "tiny.attn", 4, 2, 2);

    ASSERT_TRUE(attn != NULL);
    ASSERT_EQ(ctx->params.size, (size_t)4);
    ASSERT_STREQ(attn->Q->name, "tiny.attn.q_proj.weight");
    ASSERT_STREQ(attn->K->name, "tiny.attn.k_proj.weight");
    ASSERT_STREQ(attn->V->name, "tiny.attn.v_proj.weight");
    ASSERT_STREQ(attn->O->name, "tiny.attn.out_proj.weight");
    ASSERT_TRUE(ctx->params.data[0] == attn->Q);
    ASSERT_TRUE(ctx->params.data[1] == attn->K);
    ASSERT_TRUE(ctx->params.data[2] == attn->V);
    ASSERT_TRUE(ctx->params.data[3] == attn->O);

    pico_nn_attn_free(attn);
    pico_shutdown(ctx);
}

UTEST(self_attn, init_rejects_invalid_inputs) {
    struct PicoContext* ctx = pico_init_verbose(false);

    ASSERT_TRUE(pico_nn_attn_init(NULL, "attn", 4, 2, 2) == NULL);
    ASSERT_TRUE(pico_nn_attn_init(ctx, NULL, 4, 2, 2) == NULL);
    ASSERT_TRUE(pico_nn_attn_init(ctx, "attn", 0, 2, 2) == NULL);
    ASSERT_TRUE(pico_nn_attn_init(ctx, "attn", 4, 0, 2) == NULL);
    ASSERT_TRUE(pico_nn_attn_init(ctx, "attn", 4, 2, 0) == NULL);

    ASSERT_EQ(ctx->params.size, (size_t)0);
    pico_shutdown(ctx);
}

UTEST(self_attn, rope_rotates_qk_pairs_by_position) {
    struct PicoContext* ctx = pico_init_verbose(false);
    int64_t shape[] = {2, 4};
    float values[] = {
        1.0f, 2.0f, 3.0f, 4.0f,
        1.0f, 0.0f, 1.0f, 0.0f,
    };

    struct PicoTensor* q = pico_tensor_from_data(ctx, shape, 2, values);
    pico_nn_attn_apply_rope(q, 1, 4);

    ASSERT_NEAR_FLOAT(q->data[0], 1.0f);
    ASSERT_NEAR_FLOAT(q->data[1], 2.0f);
    ASSERT_NEAR_FLOAT(q->data[2], 3.0f);
    ASSERT_NEAR_FLOAT(q->data[3], 4.0f);

    ASSERT_NEAR_FLOAT(q->data[4], cosf(1.0f));
    ASSERT_NEAR_FLOAT(q->data[5], sinf(1.0f));
    ASSERT_NEAR_FLOAT(q->data[6], cosf(0.01f));
    ASSERT_NEAR_FLOAT(q->data[7], sinf(0.01f));

    pico_shutdown(ctx);
}
