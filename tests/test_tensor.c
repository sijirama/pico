/*
 * Tests for the tensor module.
 * These tests describe what pico_param SHOULD do; use them to drive the fixes.
 */

#include <math.h>

#include "arena.h"
#include "ops.h"
#include "tensor.h"
#include "utest.h"

// just make sure we actually get a tensor back and not null
UTEST(pico_param, returns_non_null) {
    int64_t shape[] = {2, 3};
    struct PicoTensor* t = pico_param(shape, 2);
    ASSERT_TRUE(t != NULL);
    pico_free(t);
}

// ndim and the persistent flag should be set correctly
UTEST(pico_param, metadata) {
    int64_t shape[] = {2, 3};
    struct PicoTensor* t = pico_param(shape, 2);
    ASSERT_EQ(t->ndim, 2);
    ASSERT_EQ(t->is_persistent, 1);
    pico_free(t);
}

// the shape we passed in should be stored on the tensor
UTEST(pico_param, shape_values) {
    int64_t shape[] = {2, 3};
    struct PicoTensor* t = pico_param(shape, 2);
    ASSERT_EQ(t->shape[0], (int64_t)2);
    ASSERT_EQ(t->shape[1], (int64_t)3);
    pico_free(t);
}

// row major strides, last dim is 1 and the rest are products of trailing dims
UTEST(pico_param, strides_row_major) {
    int64_t shape[] = {2, 3};
    struct PicoTensor* t = pico_param(shape, 2);
    ASSERT_EQ(t->strides[0], (int64_t)3);
    ASSERT_EQ(t->strides[1], (int64_t)1);
    pico_free(t);
}

// params need both a data buffer and a grad buffer allocated
UTEST(pico_param, allocates_data_and_grad) {
    int64_t shape[] = {2, 3};
    struct PicoTensor* t = pico_param(shape, 2);
    ASSERT_TRUE(t->data != NULL);
    ASSERT_TRUE(t->grad != NULL);
    pico_free(t);
}

// a fresh param is a leaf, so no parents and no backward fn yet
UTEST(pico_param, leaf_defaults) {
    int64_t shape[] = {2, 3};
    struct PicoTensor* t = pico_param(shape, 2);
    ASSERT_TRUE(t->parents == NULL);
    ASSERT_EQ(t->num_parents, 0);
    ASSERT_TRUE(t->_backward == NULL);
    pico_free(t);
}

// tensor should copy shape not borrow it, so mutating ours shouldnt change it
UTEST(pico_param, owns_its_shape_copy) {
    int64_t shape[] = {2, 3};
    struct PicoTensor* t = pico_param(shape, 2);
    shape[0] = 99;
    ASSERT_EQ(t->shape[0], (int64_t)2);
    pico_free(t);
}

// 1d tensor, single stride should just be 1
UTEST(pico_param, dim_1d) {
    int64_t shape[] = {5};
    struct PicoTensor* t = pico_param(shape, 1);
    ASSERT_EQ(t->ndim, 1);
    ASSERT_EQ(t->shape[0], (int64_t)5);
    ASSERT_EQ(t->strides[0], (int64_t)1);
    ASSERT_TRUE(t->data != NULL);
    ASSERT_TRUE(t->grad != NULL);
    pico_free(t);
}

// 3d strides should be products of the trailing dims
UTEST(pico_param, dim_3d) {
    int64_t shape[] = {2, 3, 4};
    struct PicoTensor* t = pico_param(shape, 3);
    ASSERT_EQ(t->ndim, 3);
    ASSERT_EQ(t->strides[0], (int64_t)12);  // 3*4
    ASSERT_EQ(t->strides[1], (int64_t)4);   // 4
    ASSERT_EQ(t->strides[2], (int64_t)1);
    ASSERT_TRUE(t->data != NULL);
    ASSERT_TRUE(t->grad != NULL);
    pico_free(t);
}

// same stride logic should still hold for 4 dims
UTEST(pico_param, dim_4d) {
    int64_t shape[] = {2, 3, 4, 5};
    struct PicoTensor* t = pico_param(shape, 4);
    ASSERT_EQ(t->ndim, 4);
    ASSERT_EQ(t->strides[0], (int64_t)60);  // 3*4*5
    ASSERT_EQ(t->strides[1], (int64_t)20);  // 4*5
    ASSERT_EQ(t->strides[2], (int64_t)5);   // 5
    ASSERT_EQ(t->strides[3], (int64_t)1);
    pico_free(t);
}

// and still hold for 5 dims, just to be sure the loop is right
UTEST(pico_param, dim_5d) {
    int64_t shape[] = {2, 3, 4, 5, 6};
    struct PicoTensor* t = pico_param(shape, 5);
    ASSERT_EQ(t->ndim, 5);
    ASSERT_EQ(t->strides[0], (int64_t)360);  // 3*4*5*6
    ASSERT_EQ(t->strides[1], (int64_t)120);  // 4*5*6
    ASSERT_EQ(t->strides[2], (int64_t)30);   // 5*6
    ASSERT_EQ(t->strides[3], (int64_t)6);    // 6
    ASSERT_EQ(t->strides[4], (int64_t)1);
    pico_free(t);
}

// freeing a null pointer should just be a safe no op, not a crash
UTEST(pico_free, null_is_safe) {
    pico_free(NULL);
    ASSERT_TRUE(1);  // reaching here without crashing is the pass
}

// alloc then free a real tensor, should run clean (run under asan to catch leaks)
UTEST(pico_free, frees_a_param) {
    int64_t shape[] = {2, 3};
    struct PicoTensor* t = pico_param(shape, 2);
    ASSERT_TRUE(t != NULL);
    pico_free(t);
    ASSERT_TRUE(1);  // no crash, and asan confirms no leak / no double free
}

// ===================================================================
//  pico_tensor_from_scalar
// ===================================================================

// a scalar tensor is a single element holding the value
UTEST(pico_tensor_from_scalar, holds_value) {
    struct Arena* ar = arena_init(4096);
    arena_ctx_push(ar);

    struct PicoTensor* s = pico_tensor_from_scalar(3.5f);
    ASSERT_TRUE(s != NULL);
    ASSERT_EQ(s->ndim, 1);
    ASSERT_EQ(s->numel, 1);
    ASSERT_TRUE(s->data[0] == 3.5f);

    arena_ctx_pop();
    arena_destroy(ar);
}

// the whole point: it broadcasts against a bigger tensor through pico_mul.
// from_scalar(2) * [1,2,3] -> [2,4,6]
UTEST(pico_tensor_from_scalar, broadcasts_through_mul) {
    struct Arena* ar = arena_init(1 << 16);
    arena_ctx_push(ar);

    int64_t s[] = {3};
    struct PicoTensor* t = pico_param(s, 1);
    t->data[0] = 1.0f;
    t->data[1] = 2.0f;
    t->data[2] = 3.0f;

    struct PicoTensor* out = pico_mul(pico_tensor_from_scalar(2.0f), t);
    ASSERT_TRUE(out != NULL);
    ASSERT_EQ(out->numel, 3);
    ASSERT_TRUE(out->data[0] == 2.0f);
    ASSERT_TRUE(out->data[1] == 4.0f);
    ASSERT_TRUE(out->data[2] == 6.0f);

    pico_free(t);
    arena_ctx_pop();
    arena_destroy(ar);
}

// ===================================================================
//  pico_rand / pico_randn  (these describe the TARGET behavior)
// ===================================================================

// pico_rand keeps the requested shape (it's just a filled tensor)
UTEST(pico_rand, keeps_shape) {
    struct Arena* ar = arena_init(4096);
    arena_ctx_push(ar);

    int64_t s[] = {2, 3};
    struct PicoTensor* t = pico_rand(ar, s, 2);

    ASSERT_TRUE(t != NULL);
    ASSERT_EQ(t->ndim, 2);
    ASSERT_EQ(t->numel, 6);
    ASSERT_TRUE(t->shape[0] == 2);
    ASSERT_TRUE(t->shape[1] == 3);

    arena_ctx_pop();
    arena_destroy(ar);
}

// pico_rand is UNIFORM on [0, 1): every element in range, and (with 1000 draws)
// there's actual spread — not a constant fill.
UTEST(pico_rand, uniform_unit_range) {
    struct Arena* ar = arena_init(1 << 16);
    arena_ctx_push(ar);

    int64_t s[] = {1000};
    struct PicoTensor* t = pico_rand(ar, s, 1);

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

    arena_ctx_pop();
    arena_destroy(ar);
}

// TARGET (currently FAILS — randn is still a uniform stub): a standard-normal
// generator must produce NEGATIVE values. uniform [0,1) never does. When randn
// is real (Box-Muller / etc.), ~half of a big sample is < 0.
UTEST(pico_randn, produces_negatives) {
    struct Arena* ar = arena_init(1 << 16);
    arena_ctx_push(ar);

    int64_t s[] = {1000};
    struct PicoTensor* t = pico_randn(ar, s, 1);

    int found_negative = 0;
    for(int64_t i = 0; i < t->numel; i++) {
        if(t->data[i] < 0.0f) {
            found_negative = 1;
            break;
        }
    }

    // tear down BEFORE asserting: this fails on purpose (randn is a uniform stub),
    // and a failed ASSERT returns early — asserting first would skip the pop and
    // leave this arena on the ctx stack, breaking later arena_ctx tests.
    arena_ctx_pop();
    arena_destroy(ar);

    ASSERT_TRUE(found_negative);  // FAILS until randn is a real normal distribution
}

// the REAL spec: randn is a STANDARD normal -> over a big sample, mean ~ 0 and
// std ~ 1. "produces negatives" alone would pass for any symmetric noise.
UTEST(pico_randn, is_standard_normal) {
    struct Arena* ar = arena_init(1 << 22);
    arena_ctx_push(ar);

    int64_t s[] = {10000};
    struct PicoTensor* t = pico_randn(ar, s, 1);

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
    arena_ctx_pop();
    arena_destroy(ar);

    ASSERT_EQ(n, 10000);                          // cat(z0,z1) reassembles the full size
    ASSERT_TRUE(mean > -0.1 && mean < 0.1);       // centered on 0
    ASSERT_TRUE(stddev > 0.85 && stddev < 1.15);  // unit variance
}
