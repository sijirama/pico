/*
 * Tests for PicoEmbedding.
 * NOTE: no UTEST_MAIN here, test_basic.c owns main + UTEST_STATE.
 */

#include "ctx.h"
#include "global.h"
#include "nn/embedding.h"
#include "tensor.h"
#include "utest.h"

UTEST(embedding, init_sets_dims_and_table_shape) {
    struct PicoContext* ctx = pico_init_verbose(false);

    struct PicoEmbedding* embedding = pico_embedding_init(ctx, 5, 3);

    ASSERT_TRUE(embedding != NULL);
    ASSERT_EQ(embedding->num_embeddings, 5);
    ASSERT_EQ(embedding->embedding_dim, 3);
    ASSERT_TRUE(embedding->table != NULL);
    ASSERT_EQ(embedding->table->ndim, 2);
    ASSERT_EQ(embedding->table->shape[0], 5);
    ASSERT_EQ(embedding->table->shape[1], 3);
    ASSERT_EQ(embedding->table->numel, 15);

    pico_shutdown(ctx);
}

UTEST(embedding, table_is_trainable_param) {
    struct PicoContext* ctx = pico_init_verbose(false);

    struct PicoEmbedding* embedding = pico_embedding_init(ctx, 4, 2);

    ASSERT_TRUE(embedding != NULL);
    ASSERT_TRUE(embedding->table != NULL);
    ASSERT_EQ(embedding->table->storage, PICO_TENSOR_STORAGE_HEAP);
    ASSERT_EQ(ctx->params.size, (size_t)1);

    pico_shutdown(ctx);
}

UTEST(embedding, apply_returns_sequence_by_embedding_dim) {
    struct PicoContext* ctx = pico_init_verbose(false);

    struct PicoEmbedding* embedding = pico_embedding_init(ctx, 4, 2);
    int64_t idx_shape[] = {3};
    float ids[] = {0.0f, 2.0f, 1.0f};
    struct PicoTensor* input = pico_tensor_from_data(ctx, idx_shape, 1, ids);

    struct PicoTensor* out = pico_embedding_apply(ctx, embedding, input);

    ASSERT_TRUE(out != NULL);
    ASSERT_EQ(out->ndim, 2);
    ASSERT_EQ(out->shape[0], 3);
    ASSERT_EQ(out->shape[1], 2);
    ASSERT_EQ(out->numel, 6);

    pico_shutdown(ctx);
}

UTEST(embedding, apply_copies_rows_from_table) {
    struct PicoContext* ctx = pico_init_verbose(false);

    struct PicoEmbedding* embedding = pico_embedding_init(ctx, 4, 2);
    float table_values[] = {
        1.0f, 2.0f,  //
        3.0f, 4.0f,  //
        5.0f, 6.0f,  //
        7.0f, 8.0f,
    };
    for(int i = 0; i < 8; i++) {
        embedding->table->data[i] = table_values[i];
    }

    int64_t idx_shape[] = {3};
    float ids[] = {2.0f, 0.0f, 3.0f};
    struct PicoTensor* input = pico_tensor_from_data(ctx, idx_shape, 1, ids);

    struct PicoTensor* out = pico_embedding_apply(ctx, embedding, input);

    ASSERT_TRUE(out != NULL);
    ASSERT_NEAR(out->data[0], 5.0f, 1e-6f);
    ASSERT_NEAR(out->data[1], 6.0f, 1e-6f);
    ASSERT_NEAR(out->data[2], 1.0f, 1e-6f);
    ASSERT_NEAR(out->data[3], 2.0f, 1e-6f);
    ASSERT_NEAR(out->data[4], 7.0f, 1e-6f);
    ASSERT_NEAR(out->data[5], 8.0f, 1e-6f);

    pico_shutdown(ctx);
}

UTEST(embedding, apply_rejects_non_1d_indices) {
    struct PicoContext* ctx = pico_init_verbose(false);

    struct PicoEmbedding* embedding = pico_embedding_init(ctx, 4, 2);
    int64_t idx_shape[] = {1, 2};
    float ids[] = {0.0f, 1.0f};
    struct PicoTensor* input = pico_tensor_from_data(ctx, idx_shape, 2, ids);

    struct PicoTensor* out = pico_embedding_apply(ctx, embedding, input);

    ASSERT_TRUE(out == NULL);

    pico_shutdown(ctx);
}

UTEST(embedding, apply_rejects_out_of_range_indices) {
    struct PicoContext* ctx = pico_init_verbose(false);

    struct PicoEmbedding* embedding = pico_embedding_init(ctx, 4, 2);
    int64_t idx_shape[] = {2};
    float ids[] = {0.0f, 4.0f};
    struct PicoTensor* input = pico_tensor_from_data(ctx, idx_shape, 1, ids);

    struct PicoTensor* out = pico_embedding_apply(ctx, embedding, input);

    ASSERT_TRUE(out == NULL);

    pico_shutdown(ctx);
}

UTEST(embedding, backward_accumulates_grad_for_repeated_indices) {
    struct PicoContext* ctx = pico_init_verbose(false);

    struct PicoEmbedding* embedding = pico_embedding_init(ctx, 3, 2);
    int64_t idx_shape[] = {3};
    float ids[] = {1.0f, 1.0f, 2.0f};
    struct PicoTensor* input = pico_tensor_from_data(ctx, idx_shape, 1, ids);
    struct PicoTensor* out = pico_embedding_apply(ctx, embedding, input);
    ASSERT_TRUE(out != NULL);

    for(int i = 0; i < 6; i++) {
        out->grad[i] = 1.0f;
    }

    pico_embedding_backward(out);

    ASSERT_NEAR(embedding->table->grad[0], 0.0f, 1e-6f);
    ASSERT_NEAR(embedding->table->grad[1], 0.0f, 1e-6f);
    ASSERT_NEAR(embedding->table->grad[2], 2.0f, 1e-6f);
    ASSERT_NEAR(embedding->table->grad[3], 2.0f, 1e-6f);
    ASSERT_NEAR(embedding->table->grad[4], 1.0f, 1e-6f);
    ASSERT_NEAR(embedding->table->grad[5], 1.0f, 1e-6f);

    pico_shutdown(ctx);
}
