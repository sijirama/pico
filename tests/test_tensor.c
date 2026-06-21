/*
 * Tests for the tensor module.
 * These tests describe what pico_param SHOULD do; use them to drive the fixes.
 */

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
