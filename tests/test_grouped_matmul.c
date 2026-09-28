/*
 * Tests for grouped batched matmul, the GQA helper:
 * A [B,Hq,M,K] @ B [B,Hkv,K,N], where kv_head = q_head / group_size.
 * NOTE: no UTEST_MAIN here, test_basic.c owns main + UTEST_STATE.
 */

#include "ctx.h"
#include "devices/backend.h"
#include "global.h"
#include "ops.h"
#include "tensor.h"
#include "utest.h"

#define ASSERT_CLOSE(got, expected) ASSERT_TRUE(((got) - (expected)) < 1e-5f && ((expected) - (got)) < 1e-5f)

UTEST(grouped_matmul, forward_maps_query_heads_to_shared_kv_heads) {
    struct PicoContext* ctx = pico_init_verbose(false);
    SimdLevel old_simd = g_simd_level;
    g_simd_level = SIMD_NONE;

    int64_t a_shape[] = {1, 4, 2, 2};
    int64_t b_shape[] = {1, 2, 2, 2};

    float a_data[] = {
        1, 2, 3, 4,
        5, 6, 7, 8,
        1, 0, 0, 1,
        2, 1, 1, 2,
    };
    float b_data[] = {
        1, 0, 0, 1,
        2, 0, 0, 3,
    };

    struct PicoTensor* a = pico_tensor_from_data(ctx, a_shape, 4, a_data);
    struct PicoTensor* b = pico_tensor_from_data(ctx, b_shape, 4, b_data);
    struct PicoTensor* out = pico_grouped_matmul(ctx, a, b, 2);

    ASSERT_TRUE(out != NULL);
    ASSERT_EQ(out->ndim, 4);
    ASSERT_EQ(out->shape[0], 1);
    ASSERT_EQ(out->shape[1], 4);
    ASSERT_EQ(out->shape[2], 2);
    ASSERT_EQ(out->shape[3], 2);

    float expected[] = {
        1, 2, 3, 4,
        5, 6, 7, 8,
        2, 0, 0, 3,
        4, 3, 2, 6,
    };

    for(int i = 0; i < 16; i++) {
        ASSERT_CLOSE(out->data[i], expected[i]);
    }

    g_simd_level = old_simd;
    pico_shutdown(ctx);
}

UTEST(grouped_matmul, wires_graph_and_stores_group_size_for_backward) {
    struct PicoContext* ctx = pico_init_verbose(false);

    int64_t a_shape[] = {1, 2, 1, 2};
    int64_t b_shape[] = {1, 1, 2, 2};
    struct PicoTensor* a = pico_param(ctx, a_shape, 4);
    struct PicoTensor* b = pico_param(ctx, b_shape, 4);
    struct PicoTensor* out = pico_grouped_matmul(ctx, a, b, 2);

    ASSERT_TRUE(out != NULL);
    ASSERT_EQ(out->num_parents, 2);
    ASSERT_TRUE(out->parents[0] == a);
    ASSERT_TRUE(out->parents[1] == b);
    ASSERT_EQ(out->op_param, 2);
    ASSERT_TRUE(out->_backward != NULL);

    pico_shutdown(ctx);
}

UTEST(grouped_matmul, backward_accumulates_shared_kv_head_grad) {
    struct PicoContext* ctx = pico_init_verbose(false);
    SimdLevel old_simd = g_simd_level;
    g_simd_level = SIMD_NONE;

    int64_t a_shape[] = {1, 2, 1, 2};
    int64_t b_shape[] = {1, 1, 2, 2};
    struct PicoTensor* a = pico_param(ctx, a_shape, 4);
    struct PicoTensor* b = pico_param(ctx, b_shape, 4);

    float a_data[] = {1, 2, 3, 4};
    float b_data[] = {5, 6, 7, 8};
    for(int i = 0; i < 4; i++) {
        a->data[i] = a_data[i];
        b->data[i] = b_data[i];
    }

    struct PicoTensor* out = pico_grouped_matmul(ctx, a, b, 2);
    ASSERT_TRUE(out != NULL);

    for(int i = 0; i < out->numel; i++) {
        out->grad[i] = 1.0f;
    }
    out->_backward(out);

    ASSERT_CLOSE(a->grad[0], 11.0f);
    ASSERT_CLOSE(a->grad[1], 15.0f);
    ASSERT_CLOSE(a->grad[2], 11.0f);
    ASSERT_CLOSE(a->grad[3], 15.0f);

    ASSERT_CLOSE(b->grad[0], 4.0f);
    ASSERT_CLOSE(b->grad[1], 4.0f);
    ASSERT_CLOSE(b->grad[2], 6.0f);
    ASSERT_CLOSE(b->grad[3], 6.0f);

    g_simd_level = old_simd;
    pico_shutdown(ctx);
}

UTEST(grouped_matmul, rejects_invalid_shapes_and_group_size) {
    struct PicoContext* ctx = pico_init_verbose(false);

    int64_t a_shape[] = {1, 4, 2, 2};
    int64_t b_shape[] = {1, 2, 2, 2};
    struct PicoTensor* a = pico_create_tensor(ctx, a_shape, 4);
    struct PicoTensor* b = pico_create_tensor(ctx, b_shape, 4);

    ASSERT_TRUE(pico_grouped_matmul(ctx, a, b, 0) == NULL);
    ASSERT_TRUE(pico_grouped_matmul(ctx, a, b, 3) == NULL);

    int64_t bad_rank_shape[] = {2, 2};
    struct PicoTensor* bad_rank = pico_create_tensor(ctx, bad_rank_shape, 2);
    ASSERT_TRUE(pico_grouped_matmul(ctx, bad_rank, b, 2) == NULL);

    pico_shutdown(ctx);
}

UTEST(grouped_matmul, rejects_cuda_backend_for_now) {
    struct PicoContext* ctx = pico_init_verbose(false);
    int64_t a_shape[] = {1, 2, 1, 2};
    int64_t b_shape[] = {1, 1, 2, 2};

    struct PicoTensor* a = pico_create_tensor_on(ctx, PICO_BACKEND_CUDA, a_shape, 4);
    struct PicoTensor* b = pico_create_tensor_on(ctx, PICO_BACKEND_CUDA, b_shape, 4);

    ASSERT_TRUE(pico_grouped_matmul(ctx, a, b, 2) == NULL);

    pico_shutdown(ctx);
}
