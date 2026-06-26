/*
 * Tests for activation functions (relu so far).
 * NOTE: no UTEST_MAIN here, test_basic.c owns main + UTEST_STATE.
 * relu is the first UNARY op (one parent) — these also check that wiring.
 */
#include "act/activations.h"
#include "arena.h"
#include "tensor.h"
#include "utest.h"

// relu(x) = max(0, x), element-wise: negatives + zero -> 0, positives pass
UTEST(act_relu, forward_clamps_negatives) {
    struct Arena* ar = arena_init(4096);
    arena_ctx_push(ar);

    int64_t s[] = {5};
    struct PicoTensor* x = pico_param(s, 1);
    x->data[0] = -2.0f;
    x->data[1] = -0.5f;
    x->data[2] = 0.0f;
    x->data[3] = 1.0f;
    x->data[4] = 3.0f;

    struct PicoTensor* out = pico_relu(x);

    ASSERT_TRUE(out->data[0] == 0.0f);
    ASSERT_TRUE(out->data[1] == 0.0f);
    ASSERT_TRUE(out->data[2] == 0.0f);
    ASSERT_TRUE(out->data[3] == 1.0f);
    ASSERT_TRUE(out->data[4] == 3.0f);

    pico_free(x);
    arena_ctx_pop();
    arena_destroy(ar);
}

// output keeps the input's shape (element-wise, no reshape)
UTEST(act_relu, forward_preserves_shape) {
    struct Arena* ar = arena_init(4096);
    arena_ctx_push(ar);

    int64_t s[] = {2, 3};
    struct PicoTensor* x = pico_param(s, 2);
    struct PicoTensor* out = pico_relu(x);

    ASSERT_EQ(out->ndim, 2);
    ASSERT_EQ(out->numel, 6);
    ASSERT_TRUE(out->shape[0] == 2);
    ASSERT_TRUE(out->shape[1] == 3);

    pico_free(x);
    arena_ctx_pop();
    arena_destroy(ar);
}

// relu is the first UNARY op: exactly one parent, backward attached
UTEST(act_relu, wires_graph_unary) {
    struct Arena* ar = arena_init(4096);
    arena_ctx_push(ar);

    int64_t s[] = {3};
    struct PicoTensor* x = pico_param(s, 1);
    struct PicoTensor* out = pico_relu(x);

    ASSERT_EQ(out->num_parents, 1);
    ASSERT_TRUE(out->parents[0] == x);
    ASSERT_TRUE(out->_backward != NULL);

    pico_free(x);
    arena_ctx_pop();
    arena_destroy(ar);
}

// backward is a GATE: grad passes where input was > 0, blocked where <= 0
UTEST(act_relu, backward_gate) {
    struct Arena* ar = arena_init(4096);
    arena_ctx_push(ar);

    int64_t s[] = {3};
    struct PicoTensor* x = pico_param(s, 1);
    x->data[0] = -2.0f;  // blocked
    x->data[1] = 0.0f;   // blocked (boundary, out=0)
    x->data[2] = 3.0f;   // passes

    struct PicoTensor* out = pico_relu(x);
    out->grad[0] = 1.0f;
    out->grad[1] = 1.0f;
    out->grad[2] = 1.0f;
    out->_backward(out);

    ASSERT_TRUE(x->grad[0] == 0.0f);
    ASSERT_TRUE(x->grad[1] == 0.0f);
    ASSERT_TRUE(x->grad[2] == 1.0f);

    pico_free(x);
    arena_ctx_pop();
    arena_destroy(ar);
}

// upstream grad is scaled, not just gated 0/1
UTEST(act_relu, backward_scales_upstream) {
    struct Arena* ar = arena_init(4096);
    arena_ctx_push(ar);

    int64_t s[] = {2};
    struct PicoTensor* x = pico_param(s, 1);
    x->data[0] = 5.0f;
    x->data[1] = -1.0f;

    struct PicoTensor* out = pico_relu(x);
    out->grad[0] = 7.0f;  // passes -> 7
    out->grad[1] = 7.0f;  // blocked -> 0
    out->_backward(out);

    ASSERT_TRUE(x->grad[0] == 7.0f);
    ASSERT_TRUE(x->grad[1] == 0.0f);

    pico_free(x);
    arena_ctx_pop();
    arena_destroy(ar);
}

// calling backward twice ACCUMULATES (+=), same rule as every other op
UTEST(act_relu, backward_accumulates) {
    struct Arena* ar = arena_init(4096);
    arena_ctx_push(ar);

    int64_t s[] = {1};
    struct PicoTensor* x = pico_param(s, 1);
    x->data[0] = 2.0f;

    struct PicoTensor* out = pico_relu(x);
    out->grad[0] = 1.0f;
    out->_backward(out);
    out->_backward(out);

    ASSERT_TRUE(x->grad[0] == 2.0f);  // 1 + 1

    pico_free(x);
    arena_ctx_pop();
    arena_destroy(ar);
}

// relu must work through the full traversal too (pico_backward seeds + walks)
UTEST(act_relu, through_pico_backward) {
    struct Arena* ar = arena_init(4096);
    arena_ctx_push(ar);

    int64_t s[] = {3};
    struct PicoTensor* x = pico_param(s, 1);
    x->data[0] = -1.0f;
    x->data[1] = 0.0f;
    x->data[2] = 2.0f;

    struct PicoTensor* out = pico_relu(x);
    pico_backward(ar, out);  // seeds out->grad=1, walks

    ASSERT_TRUE(x->grad[0] == 0.0f);  // gate closed
    ASSERT_TRUE(x->grad[1] == 0.0f);  // gate closed
    ASSERT_TRUE(x->grad[2] == 1.0f);  // gate open

    pico_free(x);
    arena_ctx_pop();
    arena_destroy(ar);
}
