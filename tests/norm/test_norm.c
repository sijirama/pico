/*
 * Tests for normalization modules.
 * NOTE: no UTEST_MAIN here, test_basic.c owns main + UTEST_STATE.
 */

#include <math.h>

#include "pico.h"
#include "utest.h"

UTEST(layernorm, init_sets_metadata) {
    struct PicoContext* ctx = pico_init_verbose(false);

    struct PicoLayerNorm* norm = pico_nn_layernorm_init(ctx, "block.norm", 3, 1e-5f);

    ASSERT_TRUE(norm != NULL);
    ASSERT_STREQ(norm->name, "block.norm");
    ASSERT_EQ(norm->normalized_dim, 3);
    ASSERT_NEAR(norm->eps, 1e-5f, 1e-8f);

    pico_nn_layernorm_free(norm);
    pico_shutdown(ctx);
}

UTEST(layernorm, init_rejects_invalid_inputs) {
    struct PicoContext* ctx = pico_init_verbose(false);

    ASSERT_TRUE(pico_nn_layernorm_init(NULL, "norm", 3, 1e-5f) == NULL);
    ASSERT_TRUE(pico_nn_layernorm_init(ctx, NULL, 3, 1e-5f) == NULL);
    ASSERT_TRUE(pico_nn_layernorm_init(ctx, "norm", 0, 1e-5f) == NULL);

    pico_shutdown(ctx);
}

UTEST(layernorm, forward_normalizes_last_dim_for_2d_tensor) {
    struct PicoContext* ctx = pico_init_verbose(false);

    int64_t shape[] = {2, 3};
    float values[] = {
        1.0f, 2.0f, 3.0f,
        2.0f, 4.0f, 6.0f,
    };
    struct PicoTensor* input = pico_tensor_from_data(ctx, shape, 2, values);
    struct PicoLayerNorm* norm = pico_nn_layernorm_init(ctx, "norm", 3, 1e-5f);

    struct PicoTensor* out = pico_nn_layernorm_forward(ctx, norm, input);

    ASSERT_TRUE(out != NULL);
    ASSERT_EQ(out->ndim, 2);
    ASSERT_EQ(out->shape[0], (int64_t)2);
    ASSERT_EQ(out->shape[1], (int64_t)3);

    ASSERT_NEAR(out->data[0], -1.2247356f, 1e-5f);
    ASSERT_NEAR(out->data[1], 0.0f, 1e-5f);
    ASSERT_NEAR(out->data[2], 1.2247356f, 1e-5f);
    ASSERT_NEAR(out->data[3], -1.2247427f, 1e-5f);
    ASSERT_NEAR(out->data[4], 0.0f, 1e-5f);
    ASSERT_NEAR(out->data[5], 1.2247427f, 1e-5f);

    pico_nn_layernorm_free(norm);
    pico_shutdown(ctx);
}

UTEST(layernorm, forward_keeps_3d_shape_and_normalizes_each_last_dim_slice) {
    struct PicoContext* ctx = pico_init_verbose(false);

    int64_t shape[] = {1, 2, 3};
    float values[] = {
        1.0f, 2.0f, 3.0f,
        4.0f, 5.0f, 6.0f,
    };
    struct PicoTensor* input = pico_tensor_from_data(ctx, shape, 3, values);
    struct PicoLayerNorm* norm = pico_nn_layernorm_init(ctx, "norm", 3, 1e-5f);

    struct PicoTensor* out = pico_nn_layernorm_forward(ctx, norm, input);

    ASSERT_TRUE(out != NULL);
    ASSERT_EQ(out->ndim, 3);
    ASSERT_EQ(out->shape[0], (int64_t)1);
    ASSERT_EQ(out->shape[1], (int64_t)2);
    ASSERT_EQ(out->shape[2], (int64_t)3);

    ASSERT_NEAR(out->data[0], -1.2247356f, 1e-5f);
    ASSERT_NEAR(out->data[1], 0.0f, 1e-5f);
    ASSERT_NEAR(out->data[2], 1.2247356f, 1e-5f);
    ASSERT_NEAR(out->data[3], -1.2247356f, 1e-5f);
    ASSERT_NEAR(out->data[4], 0.0f, 1e-5f);
    ASSERT_NEAR(out->data[5], 1.2247356f, 1e-5f);

    pico_nn_layernorm_free(norm);
    pico_shutdown(ctx);
}

UTEST(layernorm, forward_rejects_mismatched_last_dim) {
    struct PicoContext* ctx = pico_init_verbose(false);

    int64_t shape[] = {2, 3};
    struct PicoTensor* input = pico_create_tensor(ctx, shape, 2);
    struct PicoLayerNorm* norm = pico_nn_layernorm_init(ctx, "norm", 4, 1e-5f);

    ASSERT_TRUE(pico_nn_layernorm_forward(ctx, norm, input) == NULL);

    pico_nn_layernorm_free(norm);
    pico_shutdown(ctx);
}

UTEST(rmsnorm, init_sets_metadata_and_weight_param) {
    struct PicoContext* ctx = pico_init_verbose(false);

    struct PicoRMSNorm* norm = pico_nn_rmsnorm_init(ctx, "block.rms", 4, 1e-6f);

    ASSERT_TRUE(norm != NULL);
    ASSERT_STREQ(norm->name, "block.rms");
    ASSERT_EQ(norm->normalized_dim, 4);
    ASSERT_NEAR(norm->eps, 1e-6f, 1e-9f);
    ASSERT_TRUE(norm->weight != NULL);
    ASSERT_STREQ(norm->weight->name, "block.rms.weight");
    ASSERT_EQ(norm->weight->kind, PICO_TENSOR_PARAM);
    ASSERT_EQ(norm->weight->shape[0], (int64_t)4);
    for(int i = 0; i < 4; i++) {
        ASSERT_NEAR(norm->weight->data[i], 1.0f, 1e-6f);
    }

    pico_nn_rmsnorm_free(norm);
    pico_shutdown(ctx);
}

UTEST(rmsnorm, init_rejects_invalid_inputs) {
    struct PicoContext* ctx = pico_init_verbose(false);

    ASSERT_TRUE(pico_nn_rmsnorm_init(NULL, "rms", 3, 1e-5f) == NULL);
    ASSERT_TRUE(pico_nn_rmsnorm_init(ctx, NULL, 3, 1e-5f) == NULL);
    ASSERT_TRUE(pico_nn_rmsnorm_init(ctx, "rms", 0, 1e-5f) == NULL);

    pico_shutdown(ctx);
}

UTEST(rmsnorm, forward_normalizes_last_dim_for_2d_tensor) {
    struct PicoContext* ctx = pico_init_verbose(false);

    int64_t shape[] = {2, 2};
    float values[] = {
        3.0f, 4.0f,
        6.0f, 8.0f,
    };
    struct PicoTensor* input = pico_tensor_from_data(ctx, shape, 2, values);
    struct PicoRMSNorm* norm = pico_nn_rmsnorm_init(ctx, "rms", 2, 1e-12f);

    struct PicoTensor* out = pico_nn_rmsnorm_forward(ctx, norm, input);

    ASSERT_TRUE(out != NULL);
    ASSERT_EQ(out->ndim, 2);
    ASSERT_EQ(out->shape[0], (int64_t)2);
    ASSERT_EQ(out->shape[1], (int64_t)2);
    ASSERT_NEAR(out->data[0], 3.0f / sqrtf(12.5f), 1e-5f);
    ASSERT_NEAR(out->data[1], 4.0f / sqrtf(12.5f), 1e-5f);
    ASSERT_NEAR(out->data[2], 6.0f / sqrtf(50.0f), 1e-5f);
    ASSERT_NEAR(out->data[3], 8.0f / sqrtf(50.0f), 1e-5f);

    pico_nn_rmsnorm_free(norm);
    pico_shutdown(ctx);
}

UTEST(rmsnorm, forward_applies_weight) {
    struct PicoContext* ctx = pico_init_verbose(false);

    int64_t shape[] = {1, 2};
    float values[] = {3.0f, 4.0f};
    struct PicoTensor* input = pico_tensor_from_data(ctx, shape, 2, values);
    struct PicoRMSNorm* norm = pico_nn_rmsnorm_init(ctx, "rms", 2, 1e-12f);
    norm->weight->data[0] = 2.0f;
    norm->weight->data[1] = 0.5f;

    struct PicoTensor* out = pico_nn_rmsnorm_forward(ctx, norm, input);

    ASSERT_TRUE(out != NULL);
    ASSERT_NEAR(out->data[0], 2.0f * 3.0f / sqrtf(12.5f), 1e-5f);
    ASSERT_NEAR(out->data[1], 0.5f * 4.0f / sqrtf(12.5f), 1e-5f);

    pico_nn_rmsnorm_free(norm);
    pico_shutdown(ctx);
}

UTEST(rmsnorm, forward_rejects_mismatched_last_dim) {
    struct PicoContext* ctx = pico_init_verbose(false);

    int64_t shape[] = {2, 3};
    struct PicoTensor* input = pico_create_tensor(ctx, shape, 2);
    struct PicoRMSNorm* norm = pico_nn_rmsnorm_init(ctx, "rms", 4, 1e-5f);

    ASSERT_TRUE(pico_nn_rmsnorm_forward(ctx, norm, input) == NULL);

    pico_nn_rmsnorm_free(norm);
    pico_shutdown(ctx);
}

UTEST(rmsnorm, backward_populates_input_and_weight_grads) {
    struct PicoContext* ctx = pico_init_verbose(false);

    int64_t shape[] = {1, 2};
    float values[] = {3.0f, 4.0f};
    struct PicoTensor* input = pico_tensor_from_data(ctx, shape, 2, values);
    struct PicoRMSNorm* norm = pico_nn_rmsnorm_init(ctx, "rms", 2, 1e-12f);

    struct PicoTensor* out = pico_nn_rmsnorm_forward(ctx, norm, input);
    pico_backward(ctx, out);

    float rms = sqrtf(12.5f);
    float inv_rms = 1.0f / rms;
    float inv_rms_cubed = inv_rms * inv_rms * inv_rms;
    float dot = 7.0f;

    ASSERT_NEAR(input->grad[0], inv_rms - 3.0f * dot * inv_rms_cubed / 2.0f, 1e-5f);
    ASSERT_NEAR(input->grad[1], inv_rms - 4.0f * dot * inv_rms_cubed / 2.0f, 1e-5f);
    ASSERT_NEAR(norm->weight->grad[0], 3.0f / rms, 1e-5f);
    ASSERT_NEAR(norm->weight->grad[1], 4.0f / rms, 1e-5f);

    pico_nn_rmsnorm_free(norm);
    pico_shutdown(ctx);
}
