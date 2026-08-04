/*
 * Tests for the tensor module.
 * These tests describe what pico_param SHOULD do; use them to drive the fixes.
 */

#include <math.h>

#include "ctx.h"
#include "global.h"
#include "ops.h"
#include "tensor.h"
#include "utest.h"

// just make sure we actually get a tensor back and not null
UTEST(pico_param, returns_non_null) {
    struct PicoContext* ctx = pico_init_verbose(false);

    int64_t shape[] = {2, 3};
    struct PicoTensor* t = pico_param(ctx, shape, 2);
    ASSERT_TRUE(t != NULL);

    pico_shutdown(ctx);
}

// ndim and storage kind should be set correctly
UTEST(pico_param, metadata) {
    struct PicoContext* ctx = pico_init_verbose(false);

    int64_t shape[] = {2, 3};
    struct PicoTensor* t = pico_param(ctx, shape, 2);
    ASSERT_EQ(t->ndim, 2);
    ASSERT_EQ(t->storage, PICO_TENSOR_STORAGE_HEAP);

    pico_shutdown(ctx);
}

// the shape we passed in should be stored on the tensor
UTEST(pico_param, shape_values) {
    struct PicoContext* ctx = pico_init_verbose(false);

    int64_t shape[] = {2, 3};
    struct PicoTensor* t = pico_param(ctx, shape, 2);
    ASSERT_EQ(t->shape[0], (int64_t)2);
    ASSERT_EQ(t->shape[1], (int64_t)3);

    pico_shutdown(ctx);
}

// row major strides, last dim is 1 and the rest are products of trailing dims
UTEST(pico_param, strides_row_major) {
    struct PicoContext* ctx = pico_init_verbose(false);

    int64_t shape[] = {2, 3};
    struct PicoTensor* t = pico_param(ctx, shape, 2);
    ASSERT_EQ(t->strides[0], (int64_t)3);
    ASSERT_EQ(t->strides[1], (int64_t)1);

    pico_shutdown(ctx);
}

// params need both a data buffer and a grad buffer allocated
UTEST(pico_param, allocates_data_and_grad) {
    struct PicoContext* ctx = pico_init_verbose(false);

    int64_t shape[] = {2, 3};
    struct PicoTensor* t = pico_param(ctx, shape, 2);
    ASSERT_TRUE(t->data != NULL);
    ASSERT_TRUE(t->grad != NULL);

    pico_shutdown(ctx);
}

// params are registered on ctx so optimizers/modules can discover trainable
// tensors without every callsite manually threading a list around.
UTEST(pico_param, registers_on_context) {
    struct PicoContext* ctx = pico_init_verbose(false);

    int64_t shape[] = {2, 3};
    struct PicoTensor* a = pico_param(ctx, shape, 2);
    struct PicoTensor* b = pico_param(ctx, shape, 2);

    ASSERT_EQ(ctx->params.size, (size_t)2);
    ASSERT_TRUE(ctx->params.data[0] == a);
    ASSERT_TRUE(ctx->params.data[1] == b);

    pico_shutdown(ctx);
}

UTEST(pico_param, context_destroy_owns_registered_params) {
    struct PicoContext* ctx = pico_init_verbose(false);

    int64_t shape[] = {2, 3};
    struct PicoTensor* a = pico_param(ctx, shape, 2);
    struct PicoTensor* b = pico_param(ctx, shape, 2);

    ASSERT_EQ(ctx->params.size, (size_t)2);
    ASSERT_TRUE(ctx->params.data[0] == a);
    ASSERT_TRUE(ctx->params.data[1] == b);

    pico_shutdown(ctx);
}

// a fresh param is a leaf, so no parents and no backward fn yet
UTEST(pico_param, leaf_defaults) {
    struct PicoContext* ctx = pico_init_verbose(false);

    int64_t shape[] = {2, 3};
    struct PicoTensor* t = pico_param(ctx, shape, 2);
    ASSERT_TRUE(t->parents == NULL);
    ASSERT_EQ(t->num_parents, 0);
    ASSERT_TRUE(t->_backward == NULL);

    pico_shutdown(ctx);
}

// tensor should copy shape not borrow it, so mutating ours shouldnt change it
UTEST(pico_param, owns_its_shape_copy) {
    struct PicoContext* ctx = pico_init_verbose(false);

    int64_t shape[] = {2, 3};
    struct PicoTensor* t = pico_param(ctx, shape, 2);
    shape[0] = 99;
    ASSERT_EQ(t->shape[0], (int64_t)2);

    pico_shutdown(ctx);
}

// 1d tensor, single stride should just be 1
UTEST(pico_param, dim_1d) {
    struct PicoContext* ctx = pico_init_verbose(false);

    int64_t shape[] = {5};
    struct PicoTensor* t = pico_param(ctx, shape, 1);
    ASSERT_EQ(t->ndim, 1);
    ASSERT_EQ(t->shape[0], (int64_t)5);
    ASSERT_EQ(t->strides[0], (int64_t)1);
    ASSERT_TRUE(t->data != NULL);
    ASSERT_TRUE(t->grad != NULL);

    pico_shutdown(ctx);
}

// 3d strides should be products of the trailing dims
UTEST(pico_param, dim_3d) {
    struct PicoContext* ctx = pico_init_verbose(false);

    int64_t shape[] = {2, 3, 4};
    struct PicoTensor* t = pico_param(ctx, shape, 3);
    ASSERT_EQ(t->ndim, 3);
    ASSERT_EQ(t->strides[0], (int64_t)12);  // 3*4
    ASSERT_EQ(t->strides[1], (int64_t)4);   // 4
    ASSERT_EQ(t->strides[2], (int64_t)1);
    ASSERT_TRUE(t->data != NULL);
    ASSERT_TRUE(t->grad != NULL);

    pico_shutdown(ctx);
}

// same stride logic should still hold for 4 dims
UTEST(pico_param, dim_4d) {
    struct PicoContext* ctx = pico_init_verbose(false);

    int64_t shape[] = {2, 3, 4, 5};
    struct PicoTensor* t = pico_param(ctx, shape, 4);
    ASSERT_EQ(t->ndim, 4);
    ASSERT_EQ(t->strides[0], (int64_t)60);  // 3*4*5
    ASSERT_EQ(t->strides[1], (int64_t)20);  // 4*5
    ASSERT_EQ(t->strides[2], (int64_t)5);   // 5
    ASSERT_EQ(t->strides[3], (int64_t)1);

    pico_shutdown(ctx);
}

// and still hold for 5 dims, just to be sure the loop is right
UTEST(pico_param, dim_5d) {
    struct PicoContext* ctx = pico_init_verbose(false);

    int64_t shape[] = {2, 3, 4, 5, 6};
    struct PicoTensor* t = pico_param(ctx, shape, 5);
    ASSERT_EQ(t->ndim, 5);
    ASSERT_EQ(t->strides[0], (int64_t)360);  // 3*4*5*6
    ASSERT_EQ(t->strides[1], (int64_t)120);  // 4*5*6
    ASSERT_EQ(t->strides[2], (int64_t)30);   // 5*6
    ASSERT_EQ(t->strides[3], (int64_t)6);    // 6
    ASSERT_EQ(t->strides[4], (int64_t)1);

    pico_shutdown(ctx);
}

// ===================================================================
//  pico_tensor_from_scalar
// ===================================================================

// a scalar tensor is a single element holding the value
UTEST(pico_tensor_from_scalar, holds_value) {
    struct PicoContext* ctx = pico_init_verbose(false);

    struct PicoTensor* s = pico_tensor_from_scalar(ctx, 3.5f);
    ASSERT_TRUE(s != NULL);
    ASSERT_EQ(s->ndim, 1);
    ASSERT_EQ(s->numel, 1);
    ASSERT_TRUE(s->data[0] == 3.5f);

    pico_shutdown(ctx);
}

// the whole point: it broadcasts against a bigger tensor through pico_mul.
// from_scalar(2) * [1,2,3] -> [2,4,6]
UTEST(pico_tensor_from_scalar, broadcasts_through_mul) {
    struct PicoContext* ctx = pico_init_verbose(false);

    int64_t s[] = {3};
    struct PicoTensor* t = pico_param(ctx, s, 1);
    t->data[0] = 1.0f;
    t->data[1] = 2.0f;
    t->data[2] = 3.0f;

    struct PicoTensor* out = pico_mul(ctx, pico_tensor_from_scalar(ctx, 2.0f), t);
    ASSERT_TRUE(out != NULL);
    ASSERT_EQ(out->numel, 3);
    ASSERT_TRUE(out->data[0] == 2.0f);
    ASSERT_TRUE(out->data[1] == 4.0f);
    ASSERT_TRUE(out->data[2] == 6.0f);

    pico_shutdown(ctx);
}

// ===================================================================
//  pico_tensor_from_data
// ===================================================================

UTEST(pico_tensor_from_data, copies_values_and_metadata) {
    struct PicoContext* ctx = pico_init_verbose(false);

    int64_t shape[] = {2, 3};
    float data[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};

    struct PicoTensor* t = pico_tensor_from_data(ctx, shape, 2, data);
    ASSERT_TRUE(t != NULL);
    ASSERT_EQ(t->ndim, 2);
    ASSERT_EQ(t->numel, 6);
    ASSERT_EQ(t->shape[0], (int64_t)2);
    ASSERT_EQ(t->shape[1], (int64_t)3);
    ASSERT_EQ(t->strides[0], (int64_t)3);
    ASSERT_EQ(t->strides[1], (int64_t)1);
    ASSERT_TRUE(t->data[0] == 1.0f);
    ASSERT_TRUE(t->data[5] == 6.0f);

    pico_shutdown(ctx);
}

UTEST(pico_tensor_from_data, owns_a_copy_of_input_data) {
    struct PicoContext* ctx = pico_init_verbose(false);

    int64_t shape[] = {3};
    float data[] = {1.0f, 2.0f, 3.0f};

    struct PicoTensor* t = pico_tensor_from_data(ctx, shape, 1, data);
    ASSERT_TRUE(t != NULL);
    data[0] = 99.0f;
    ASSERT_TRUE(t->data[0] == 1.0f);

    pico_shutdown(ctx);
}

UTEST(pico_tensor_from_data, rejects_null_data) {
    struct PicoContext* ctx = pico_init_verbose(false);

    int64_t shape[] = {3};
    struct PicoTensor* t = pico_tensor_from_data(ctx, shape, 1, NULL);
    ASSERT_TRUE(t == NULL);

    pico_shutdown(ctx);
}

// ===================================================================
//  pico_rand / pico_randn  (these describe the TARGET behavior)
// ===================================================================

// pico_rand keeps the requested shape (it's just a filled tensor)
UTEST(pico_rand, keeps_shape) {
    struct PicoContext* ctx = pico_init_verbose(false);

    int64_t s[] = {2, 3};
    struct PicoTensor* t = pico_rand(ctx, s, 2);

    ASSERT_TRUE(t != NULL);
    ASSERT_EQ(t->ndim, 2);
    ASSERT_EQ(t->numel, 6);
    ASSERT_TRUE(t->shape[0] == 2);
    ASSERT_TRUE(t->shape[1] == 3);

    pico_shutdown(ctx);
}

// pico_rand is UNIFORM on [0, 1): every element in range, and (with 1000 draws)
// there's actual spread — not a constant fill.
UTEST(pico_rand, uniform_unit_range) {
    struct PicoContext* ctx = pico_init_verbose(false);

    int64_t s[] = {1000};
    struct PicoTensor* t = pico_rand(ctx, s, 1);

    float lo = 2.0f, hi = -1.0f;
    for(int64_t i = 0; i < t->numel; i++) {
        ASSERT_TRUE(t->data[i] >= 0.0f);
        ASSERT_TRUE(t->data[i] < 1.0f);
        if(t->data[i] < lo)
            lo = t->data[i];
        if(t->data[i] > hi)
            hi = t->data[i];
    }
    ASSERT_TRUE(hi > lo);  // there's variation, not a constant

    pico_shutdown(ctx);
}

// randn should produce negative values. uniform [0,1) never does, while a real
// normal distribution puts roughly half the sample below 0.
UTEST(pico_randn, produces_negatives) {
    struct PicoContext* ctx = pico_init_verbose(false);

    int64_t s[] = {1000};
    struct PicoTensor* t = pico_randn(ctx, s, 1);

    int found_negative = 0;
    for(int64_t i = 0; i < t->numel; i++) {
        if(t->data[i] < 0.0f) {
            found_negative = 1;
            break;
        }
    }

    pico_shutdown(ctx);

    ASSERT_TRUE(found_negative);
}

// the REAL spec: randn is a STANDARD normal -> over a big sample, mean ~ 0 and
// std ~ 1. "produces negatives" alone would pass for any symmetric noise.
UTEST(pico_randn, is_standard_normal) {
    struct PicoContext* ctx = pico_init_verbose(false);

    int64_t s[] = {10000};
    struct PicoTensor* t = pico_randn(ctx, s, 1);

    double sum = 0.0;
    for(int64_t i = 0; i < t->numel; i++) sum += t->data[i];
    double mean = sum / (double)t->numel;

    double sq = 0.0;
    for(int64_t i = 0; i < t->numel; i++) {
        double d = t->data[i] - mean;
        sq += d * d;
    }
    double stddev = sqrt(sq / (double)t->numel);

    int64_t n = t->numel;

    // capture + teardown before asserting (fail-safe for the ctx stack)
    pico_shutdown(ctx);

    ASSERT_EQ(n, 10000);
    ASSERT_TRUE(mean > -0.1 && mean < 0.1);       // centered on 0
    ASSERT_TRUE(stddev > 0.85 && stddev < 1.15);  // unit variance
}

UTEST(pico_randn, keeps_odd_1d_shape) {
    struct PicoContext* ctx = pico_init_verbose(false);

    int64_t s[] = {5};
    struct PicoTensor* t = pico_randn(ctx, s, 1);

    ASSERT_TRUE(t != NULL);
    ASSERT_EQ(t->ndim, 1);
    ASSERT_EQ(t->numel, 5);
    ASSERT_EQ(t->shape[0], (int64_t)5);

    pico_shutdown(ctx);
}

UTEST(pico_randn, keeps_multidim_shape) {
    struct PicoContext* ctx = pico_init_verbose(false);

    int64_t s[] = {2, 4};
    struct PicoTensor* t = pico_randn(ctx, s, 2);

    ASSERT_TRUE(t != NULL);
    ASSERT_EQ(t->ndim, 2);
    ASSERT_EQ(t->numel, 8);
    ASSERT_EQ(t->shape[0], (int64_t)2);
    ASSERT_EQ(t->shape[1], (int64_t)4);
    ASSERT_EQ(t->strides[0], (int64_t)4);
    ASSERT_EQ(t->strides[1], (int64_t)1);

    pico_shutdown(ctx);
}

UTEST(pico_randn, values_are_finite) {
    struct PicoContext* ctx = pico_init_verbose(false);

    int64_t s[] = {1000};
    struct PicoTensor* t = pico_randn(ctx, s, 1);

    for(int64_t i = 0; i < t->numel; i++) {
        ASSERT_TRUE(isfinite(t->data[i]));
    }

    pico_shutdown(ctx);
}
