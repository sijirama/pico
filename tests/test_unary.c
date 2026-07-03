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

// ===================================================================
//  UNARY BACKWARDS — PUNCH-LIST. each asserts x.grad = upstream * f'(x).
//  upstream grad is 2 (not 1) so a backward that forgets to multiply by
//  self->grad also fails. red until the bodies in autograd.h are filled;
//  capture -> teardown -> assert so a failure doesn't leak the ctx stack.
// ===================================================================

// sqrt'(x) = 1/(2*sqrt(x)).  x=4 -> 1/4 = 0.25 ; * upstream 2 = 0.5
UTEST(unary_backward, sqrt) {
    struct Arena* ar = arena_init(4096);
    arena_ctx_push(ar);

    int64_t s[] = {1};
    struct PicoTensor* x = pico_param(s, 1);
    x->data[0] = 4.0f;

    struct PicoTensor* out = pico_tensor_sqrt(x);
    out->grad[0] = 2.0f;  // upstream
    out->_backward(out);
    float gx = x->grad[0];

    pico_free(x);
    arena_ctx_pop();
    arena_destroy(ar);

    ASSERT_TRUE(NEAR(gx, 0.5f));
}

// sin'(x) = cos(x).  x=pi -> cos(pi) = -1 ; * 2 = -2.
// (x=pi, not 0: at 0, sin(0)=0 so cos(output)==cos(input) and the input-vs-output
// bug hides. at pi it's exposed — cos(sin(pi)) ~ 1, but cos(pi) = -1.)
UTEST(unary_backward, sin) {
    struct Arena* ar = arena_init(4096);
    arena_ctx_push(ar);

    int64_t s[] = {1};
    struct PicoTensor* x = pico_param(s, 1);
    x->data[0] = PI_F;

    struct PicoTensor* out = pico_tensor_sin(x);
    out->grad[0] = 2.0f;
    out->_backward(out);
    float gx = x->grad[0];

    pico_free(x);
    arena_ctx_pop();
    arena_destroy(ar);

    ASSERT_TRUE(NEAR(gx, -2.0f));
}

// cos'(x) = -sin(x).  x=pi/2 -> -1 ; * 2 = -2  (x=0 would be 0, can't tell from a no-op)
UTEST(unary_backward, cos) {
    struct Arena* ar = arena_init(4096);
    arena_ctx_push(ar);

    int64_t s[] = {1};
    struct PicoTensor* x = pico_param(s, 1);
    x->data[0] = PI_F / 2.0f;

    struct PicoTensor* out = pico_tensor_cos(x);
    out->grad[0] = 2.0f;
    out->_backward(out);
    float gx = x->grad[0];

    pico_free(x);
    arena_ctx_pop();
    arena_destroy(ar);

    ASSERT_TRUE(NEAR(gx, -2.0f));
}

// tan'(x) = sec^2(x) = 1/cos^2(x).  x=0 -> 1 ; * 2 = 2
UTEST(unary_backward, tan) {
    struct Arena* ar = arena_init(4096);
    arena_ctx_push(ar);

    int64_t s[] = {1};
    struct PicoTensor* x = pico_param(s, 1);
    x->data[0] = 0.0f;

    struct PicoTensor* out = pico_tensor_tan(x);
    out->grad[0] = 2.0f;
    out->_backward(out);
    float gx = x->grad[0];

    pico_free(x);
    arena_ctx_pop();
    arena_destroy(ar);

    ASSERT_TRUE(NEAR(gx, 2.0f));
}

// tanh'(x) = 1 - tanh^2(x).  x=0 -> 1 ; * 2 = 2
UTEST(unary_backward, tanh) {
    struct Arena* ar = arena_init(4096);
    arena_ctx_push(ar);

    int64_t s[] = {1};
    struct PicoTensor* x = pico_param(s, 1);
    x->data[0] = 0.0f;

    struct PicoTensor* out = pico_tensor_tanh(x);
    out->grad[0] = 2.0f;
    out->_backward(out);
    float gx = x->grad[0];

    pico_free(x);
    arena_ctx_pop();
    arena_destroy(ar);

    ASSERT_TRUE(NEAR(gx, 2.0f));
}

// log'(x) = 1/x.  x=2 -> 0.5 ; * 2 = 1
UTEST(unary_backward, log) {
    struct Arena* ar = arena_init(4096);
    arena_ctx_push(ar);

    int64_t s[] = {1};
    struct PicoTensor* x = pico_param(s, 1);
    x->data[0] = 2.0f;

    struct PicoTensor* out = pico_tensor_log(x);
    out->grad[0] = 2.0f;
    out->_backward(out);
    float gx = x->grad[0];

    pico_free(x);
    arena_ctx_pop();
    arena_destroy(ar);

    ASSERT_TRUE(NEAR(gx, 1.0f));
}
