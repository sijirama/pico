/*
 * Tests for the unary element-wise math ops (pico_tensor_sqrt/sin/cos/tan/tanh).
 * FORWARD ONLY for now — backwards are TODO, so there's one punch-list test
 * (unary_backward_is_todo) that stays red until the _backward fns are wired.
 * NOTE: no UTEST_MAIN here, test_basic.c owns main + UTEST_STATE.
 */
#include <math.h>

#include "arena.h"
#include "ops.h"
#include "tensor.h"
#include "utest.h"

#define NEAR(a, b) (fabsf((a) - (b)) < 1e-5f)
#define PI_F 3.14159265358979323846f  // M_PI isn't exposed under -std=c11

// sqrt of perfect squares is exact -> can use ==
UTEST(unary, sqrt_forward) {
    struct Arena* ar = arena_init(4096);
    arena_ctx_push(ar);

    int64_t s[] = {4};
    struct PicoTensor* x = pico_param(s, 1);
    x->data[0] = 0.0f;
    x->data[1] = 4.0f;
    x->data[2] = 9.0f;
    x->data[3] = 16.0f;

    struct PicoTensor* out = pico_tensor_sqrt(x);
    ASSERT_TRUE(out != NULL);
    ASSERT_TRUE(out->data[0] == 0.0f);
    ASSERT_TRUE(out->data[1] == 2.0f);
    ASSERT_TRUE(out->data[2] == 3.0f);
    ASSERT_TRUE(out->data[3] == 4.0f);

    pico_free(x);
    arena_ctx_pop();
    arena_destroy(ar);
}

// sin(0)=0, sin(pi/2)=1
UTEST(unary, sin_forward) {
    struct Arena* ar = arena_init(4096);
    arena_ctx_push(ar);

    int64_t s[] = {2};
    struct PicoTensor* x = pico_param(s, 1);
    x->data[0] = 0.0f;
    x->data[1] = PI_F / 2.0f;

    struct PicoTensor* out = pico_tensor_sin(x);
    ASSERT_TRUE(NEAR(out->data[0], 0.0f));
    ASSERT_TRUE(NEAR(out->data[1], 1.0f));

    pico_free(x);
    arena_ctx_pop();
    arena_destroy(ar);
}

// cos(0)=1, cos(pi)=-1
UTEST(unary, cos_forward) {
    struct Arena* ar = arena_init(4096);
    arena_ctx_push(ar);

    int64_t s[] = {2};
    struct PicoTensor* x = pico_param(s, 1);
    x->data[0] = 0.0f;
    x->data[1] = PI_F;

    struct PicoTensor* out = pico_tensor_cos(x);
    ASSERT_TRUE(NEAR(out->data[0], 1.0f));
    ASSERT_TRUE(NEAR(out->data[1], -1.0f));

    pico_free(x);
    arena_ctx_pop();
    arena_destroy(ar);
}

// tan(0)=0, tan(pi/4)=1
UTEST(unary, tan_forward) {
    struct Arena* ar = arena_init(4096);
    arena_ctx_push(ar);

    int64_t s[] = {2};
    struct PicoTensor* x = pico_param(s, 1);
    x->data[0] = 0.0f;
    x->data[1] = PI_F / 4.0f;

    struct PicoTensor* out = pico_tensor_tan(x);
    ASSERT_TRUE(NEAR(out->data[0], 0.0f));
    ASSERT_TRUE(NEAR(out->data[1], 1.0f));

    pico_free(x);
    arena_ctx_pop();
    arena_destroy(ar);
}

// tanh(0)=0, saturates toward 1 for large positive input
UTEST(unary, tanh_forward) {
    struct Arena* ar = arena_init(4096);
    arena_ctx_push(ar);

    int64_t s[] = {2};
    struct PicoTensor* x = pico_param(s, 1);
    x->data[0] = 0.0f;
    x->data[1] = 20.0f;

    struct PicoTensor* out = pico_tensor_tanh(x);
    ASSERT_TRUE(NEAR(out->data[0], 0.0f));
    ASSERT_TRUE(NEAR(out->data[1], 1.0f));

    pico_free(x);
    arena_ctx_pop();
    arena_destroy(ar);
}

// natural log: log(1)=0 (exact), log(e)=1
UTEST(unary, log_forward) {
    struct Arena* ar = arena_init(4096);
    arena_ctx_push(ar);

    int64_t s[] = {2};
    struct PicoTensor* x = pico_param(s, 1);
    x->data[0] = 1.0f;
    x->data[1] = 2.71828182845904523536f;  // e

    struct PicoTensor* out = pico_tensor_log(x);
    ASSERT_TRUE(out->data[0] == 0.0f);
    ASSERT_TRUE(NEAR(out->data[1], 1.0f));

    pico_free(x);
    arena_ctx_pop();
    arena_destroy(ar);
}

// output keeps the input shape and wires the single parent (unary op)
UTEST(unary, preserves_shape_and_wires_parent) {
    struct Arena* ar = arena_init(4096);
    arena_ctx_push(ar);

    int64_t s[] = {2, 3};
    struct PicoTensor* x = pico_param(s, 2);
    struct PicoTensor* out = pico_tensor_sin(x);

    ASSERT_EQ(out->ndim, 2);
    ASSERT_EQ(out->numel, 6);
    ASSERT_TRUE(out->shape[0] == 2);
    ASSERT_TRUE(out->shape[1] == 3);
    ASSERT_EQ(out->num_parents, 1);
    ASSERT_TRUE(out->parents[0] == x);

    pico_free(x);
    arena_ctx_pop();
    arena_destroy(ar);
}

// PUNCH-LIST (fails now): forward-only, so _backward is NULL. this stays red until
// the unary backwards (sin'=cos, cos'=-sin, tan'=sec^2, tanh'=1-tanh^2,
// sqrt'=1/(2*sqrt)) are wired. same for cos/tan/tanh/sqrt — sin stands in for all.
UTEST(unary, backward_is_todo) {
    struct Arena* ar = arena_init(4096);
    arena_ctx_push(ar);

    int64_t s[] = {2};
    struct PicoTensor* x = pico_param(s, 1);
    struct PicoTensor* out = pico_tensor_sin(x);
    int has_backward = (out->_backward != NULL);

    // tear down BEFORE asserting: this test fails on purpose, and utest returns
    // on a failed ASSERT — asserting first would skip the pop and leave this arena
    // on the ctx stack, polluting later arena_ctx tests.
    pico_free(x);
    arena_ctx_pop();
    arena_destroy(ar);

    ASSERT_TRUE(has_backward);  // TODO: wire the unary backwards
}
