/*
 * Tests for activation functions (relu so far).
 * NOTE: no UTEST_MAIN here, test_basic.c owns main + UTEST_STATE.
 * relu is the first UNARY op (one parent) — these also check that wiring.
 */
#include <math.h>

#include "act/activations.h"
#include "ctx.h"
#include "global.h"
#include "tensor.h"
#include "utest.h"

// relu(x) = max(0, x), element-wise: negatives + zero -> 0, positives pass
UTEST(act_relu, forward_clamps_negatives) {
    struct PicoContext* ctx = pico_init_verbose(false);

    int64_t s[] = {5};
    struct PicoTensor* x = pico_param(ctx, s, 1);
    x->data[0] = -2.0f;
    x->data[1] = -0.5f;
    x->data[2] = 0.0f;
    x->data[3] = 1.0f;
    x->data[4] = 3.0f;

    struct PicoTensor* out = pico_relu(ctx, x);

    ASSERT_TRUE(out->data[0] == 0.0f);
    ASSERT_TRUE(out->data[1] == 0.0f);
    ASSERT_TRUE(out->data[2] == 0.0f);
    ASSERT_TRUE(out->data[3] == 1.0f);
    ASSERT_TRUE(out->data[4] == 3.0f);

    pico_shutdown(ctx);
}

// output keeps the input's shape (element-wise, no reshape)
UTEST(act_relu, forward_preserves_shape) {
    struct PicoContext* ctx = pico_init_verbose(false);

    int64_t s[] = {2, 3};
    struct PicoTensor* x = pico_param(ctx, s, 2);
    struct PicoTensor* out = pico_relu(ctx, x);

    ASSERT_EQ(out->ndim, 2);
    ASSERT_EQ(out->numel, 6);
    ASSERT_TRUE(out->shape[0] == 2);
    ASSERT_TRUE(out->shape[1] == 3);

    pico_shutdown(ctx);
}

// relu is the first UNARY op: exactly one parent, backward attached
UTEST(act_relu, wires_graph_unary) {
    struct PicoContext* ctx = pico_init_verbose(false);

    int64_t s[] = {3};
    struct PicoTensor* x = pico_param(ctx, s, 1);
    struct PicoTensor* out = pico_relu(ctx, x);

    ASSERT_EQ(out->num_parents, 1);
    ASSERT_TRUE(out->parents[0] == x);
    ASSERT_TRUE(out->_backward != NULL);

    pico_shutdown(ctx);
}

// backward is a GATE: grad passes where input was > 0, blocked where <= 0
UTEST(act_relu, backward_gate) {
    struct PicoContext* ctx = pico_init_verbose(false);

    int64_t s[] = {3};
    struct PicoTensor* x = pico_param(ctx, s, 1);
    x->data[0] = -2.0f;  // blocked
    x->data[1] = 0.0f;   // blocked (boundary, out=0)
    x->data[2] = 3.0f;   // passes

    struct PicoTensor* out = pico_relu(ctx, x);
    out->grad[0] = 1.0f;
    out->grad[1] = 1.0f;
    out->grad[2] = 1.0f;
    out->_backward(out);

    ASSERT_TRUE(x->grad[0] == 0.0f);
    ASSERT_TRUE(x->grad[1] == 0.0f);
    ASSERT_TRUE(x->grad[2] == 1.0f);

    pico_shutdown(ctx);
}

// upstream grad is scaled, not just gated 0/1
UTEST(act_relu, backward_scales_upstream) {
    struct PicoContext* ctx = pico_init_verbose(false);

    int64_t s[] = {2};
    struct PicoTensor* x = pico_param(ctx, s, 1);
    x->data[0] = 5.0f;
    x->data[1] = -1.0f;

    struct PicoTensor* out = pico_relu(ctx, x);
    out->grad[0] = 7.0f;  // passes -> 7
    out->grad[1] = 7.0f;  // blocked -> 0
    out->_backward(out);

    ASSERT_TRUE(x->grad[0] == 7.0f);
    ASSERT_TRUE(x->grad[1] == 0.0f);

    pico_shutdown(ctx);
}

// calling backward twice ACCUMULATES (+=), same rule as every other op
UTEST(act_relu, backward_accumulates) {
    struct PicoContext* ctx = pico_init_verbose(false);

    int64_t s[] = {1};
    struct PicoTensor* x = pico_param(ctx, s, 1);
    x->data[0] = 2.0f;

    struct PicoTensor* out = pico_relu(ctx, x);
    out->grad[0] = 1.0f;
    out->_backward(out);
    out->_backward(out);

    ASSERT_TRUE(x->grad[0] == 2.0f);  // 1 + 1

    pico_shutdown(ctx);
}

// relu must work through the full traversal too (pico_backward seeds + walks)
UTEST(act_relu, through_pico_backward) {
    struct PicoContext* ctx = pico_init_verbose(false);

    int64_t s[] = {3};
    struct PicoTensor* x = pico_param(ctx, s, 1);
    x->data[0] = -1.0f;
    x->data[1] = 0.0f;
    x->data[2] = 2.0f;

    struct PicoTensor* out = pico_relu(ctx, x);
    pico_backward(ctx, out);  // seeds out->grad=1, walks

    ASSERT_TRUE(x->grad[0] == 0.0f);  // gate closed
    ASSERT_TRUE(x->grad[1] == 0.0f);  // gate closed
    ASSERT_TRUE(x->grad[2] == 1.0f);  // gate open

    pico_shutdown(ctx);
}

UTEST(act_sigmoid, forward_values) {
    struct PicoContext* ctx = pico_init_verbose(false);

    int64_t s[] = {3};
    struct PicoTensor* x = pico_param(ctx, s, 1);
    x->data[0] = -1.0f;
    x->data[1] = 0.0f;
    x->data[2] = 1.0f;

    struct PicoTensor* out = pico_sigmoid(ctx, x);

    ASSERT_NEAR(out->data[0], 1.0f / (1.0f + expf(1.0f)), 1e-6f);
    ASSERT_NEAR(out->data[1], 0.5f, 1e-6f);
    ASSERT_NEAR(out->data[2], 1.0f / (1.0f + expf(-1.0f)), 1e-6f);

    pico_shutdown(ctx);
}

UTEST(act_sigmoid, backward_uses_output_derivative) {
    struct PicoContext* ctx = pico_init_verbose(false);

    int64_t s[] = {1};
    struct PicoTensor* x = pico_param(ctx, s, 1);
    x->data[0] = 0.0f;

    struct PicoTensor* out = pico_sigmoid(ctx, x);
    out->grad[0] = 4.0f;
    out->_backward(out);

    ASSERT_NEAR(x->grad[0], 1.0f, 1e-6f);

    pico_shutdown(ctx);
}

UTEST(act_swiglu, forward_silu_times_gate) {
    struct PicoContext* ctx = pico_init_verbose(false);

    int64_t s[] = {3};
    struct PicoTensor* x = pico_param(ctx, s, 1);
    struct PicoTensor* gate = pico_param(ctx, s, 1);
    x->data[0] = -1.0f;
    x->data[1] = 0.0f;
    x->data[2] = 2.0f;
    gate->data[0] = 3.0f;
    gate->data[1] = 4.0f;
    gate->data[2] = 5.0f;

    struct PicoTensor* out = pico_swiglu(ctx, x, gate);

    ASSERT_TRUE(out != NULL);
    ASSERT_NEAR(out->data[0], silu(-1.0f) * 3.0f, 1e-6f);
    ASSERT_NEAR(out->data[1], 0.0f, 1e-6f);
    ASSERT_NEAR(out->data[2], silu(2.0f) * 5.0f, 1e-6f);

    pico_shutdown(ctx);
}

UTEST(act_swiglu, wires_two_parent_graph) {
    struct PicoContext* ctx = pico_init_verbose(false);

    int64_t s[] = {2};
    struct PicoTensor* x = pico_param(ctx, s, 1);
    struct PicoTensor* gate = pico_param(ctx, s, 1);

    struct PicoTensor* out = pico_swiglu(ctx, x, gate);

    ASSERT_EQ(out->num_parents, 2);
    ASSERT_TRUE(out->parents[0] == x);
    ASSERT_TRUE(out->parents[1] == gate);
    ASSERT_TRUE(out->_backward != NULL);

    pico_shutdown(ctx);
}

UTEST(act_swiglu, backward_populates_x_and_gate_grads) {
    struct PicoContext* ctx = pico_init_verbose(false);

    int64_t s[] = {2};
    struct PicoTensor* x = pico_param(ctx, s, 1);
    struct PicoTensor* gate = pico_param(ctx, s, 1);
    x->data[0] = 0.0f;
    x->data[1] = 2.0f;
    gate->data[0] = 3.0f;
    gate->data[1] = 5.0f;

    struct PicoTensor* out = pico_swiglu(ctx, x, gate);
    out->grad[0] = 2.0f;
    out->grad[1] = 4.0f;
    out->_backward(out);

    float sig2 = sigmoid(2.0f);
    float silu2_grad = sig2 + 2.0f * sig2 * (1.0f - sig2);
    ASSERT_NEAR(x->grad[0], 2.0f * 3.0f * 0.5f, 1e-6f);
    ASSERT_NEAR(x->grad[1], 4.0f * 5.0f * silu2_grad, 1e-5f);
    ASSERT_NEAR(gate->grad[0], 0.0f, 1e-6f);
    ASSERT_NEAR(gate->grad[1], 4.0f * silu(2.0f), 1e-5f);

    pico_shutdown(ctx);
}

UTEST(act_swiglu, rejects_shape_mismatch) {
    struct PicoContext* ctx = pico_init_verbose(false);

    int64_t sx[] = {2};
    int64_t sg[] = {3};
    struct PicoTensor* x = pico_param(ctx, sx, 1);
    struct PicoTensor* gate = pico_param(ctx, sg, 1);

    ASSERT_TRUE(pico_swiglu(ctx, x, gate) == NULL);

    pico_shutdown(ctx);
}
