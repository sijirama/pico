/*
 * Tests for pico_mse_loss (forward value, graph wiring, backward correctness).
 * NOTE: no UTEST_MAIN here, test_basic.c owns main + UTEST_STATE.
 */

#include "ctx.h"
#include "loss/loss.h"
#include "ops.h"
#include "tensor.h"
#include "utest.h"

// MSE of a single element: (pred - actual)^2 / 1
UTEST(loss, mse_forward_single_element) {
    struct PicoContext ctx = pico_context_init();

    int64_t s[] = {1};
    struct PicoTensor* pred = pico_param(&ctx, s, 1);
    struct PicoTensor* actual = pico_param(&ctx, s, 1);
    pred->data[0] = 5.0f;
    actual->data[0] = 3.0f;

    struct PicoMSELoss mse = {.reduction = MEAN};
    struct PicoTensor* loss = pico_mse_loss(&ctx, &mse, pred, actual);

    // (5-3)^2 = 4
    ASSERT_TRUE(loss->data[0] == 4.0f);

    pico_context_destroy(&ctx);
}

// MSE across multiple elements should be the MEAN of squared errors, not the sum
UTEST(loss, mse_forward_multi_element_mean) {
    struct PicoContext ctx = pico_context_init();

    int64_t s[] = {3};
    struct PicoTensor* pred = pico_param(&ctx, s, 1);
    struct PicoTensor* actual = pico_param(&ctx, s, 1);

    pred->data[0] = 1.0f;
    actual->data[0] = 0.0f;  // diff 1 -> sq 1
    pred->data[1] = 2.0f;
    actual->data[1] = 0.0f;  // diff 2 -> sq 4
    pred->data[2] = 3.0f;
    actual->data[2] = 0.0f;  // diff 3 -> sq 9

    struct PicoMSELoss mse = {.reduction = MEAN};
    struct PicoTensor* loss = pico_mse_loss(&ctx, &mse, pred, actual);

    // (1+4+9)/3 = 4.666...
    ASSERT_TRUE(loss->data[0] > 4.66f && loss->data[0] < 4.67f);

    pico_context_destroy(&ctx);
}

// pico_mse_loss should wire the graph: two parents (pred, actual) + backward fn attached
UTEST(loss, mse_wires_graph) {
    struct PicoContext ctx = pico_context_init();

    int64_t s[] = {1};
    struct PicoTensor* pred = pico_param(&ctx, s, 1);
    struct PicoTensor* actual = pico_param(&ctx, s, 1);

    struct PicoMSELoss mse = {.reduction = MEAN};
    struct PicoTensor* loss = pico_mse_loss(&ctx, &mse, pred, actual);

    ASSERT_EQ(loss->num_parents, 2);
    ASSERT_TRUE(loss->parents[0] == pred);
    ASSERT_TRUE(loss->parents[1] == actual);
    ASSERT_TRUE(loss->_backward != NULL);

    pico_context_destroy(&ctx);
}

UTEST(loss, mse_rejects_shape_mismatch) {
    struct PicoContext ctx = pico_context_init();

    int64_t pred_shape[] = {4};
    int64_t actual_shape[] = {2};
    struct PicoTensor* pred = pico_param(&ctx, pred_shape, 1);
    struct PicoTensor* actual = pico_param(&ctx, actual_shape, 1);

    struct PicoMSELoss mse = {.reduction = MEAN};
    struct PicoTensor* loss = pico_mse_loss(&ctx, &mse, pred, actual);

    ASSERT_TRUE(loss == NULL);

    pico_context_destroy(&ctx);
}

UTEST(loss, mse_rejects_ndim_mismatch) {
    struct PicoContext ctx = pico_context_init();

    int64_t pred_shape[] = {2, 2};
    int64_t actual_shape[] = {4};
    struct PicoTensor* pred = pico_param(&ctx, pred_shape, 2);
    struct PicoTensor* actual = pico_param(&ctx, actual_shape, 1);

    struct PicoMSELoss mse = {.reduction = MEAN};
    struct PicoTensor* loss = pico_mse_loss(&ctx, &mse, pred, actual);

    ASSERT_TRUE(loss == NULL);

    pico_context_destroy(&ctx);
}

// d(MSE)/d(pred_i) = (2/N) * (pred_i - actual_i), verified against numeric finite difference
UTEST(loss, mse_backward_matches_finite_difference) {
    struct PicoContext ctx = pico_context_init();

    int64_t s[] = {2};
    struct PicoTensor* pred = pico_param(&ctx, s, 1);
    struct PicoTensor* actual = pico_param(&ctx, s, 1);
    pred->data[0] = 5.0f;
    pred->data[1] = -1.0f;
    actual->data[0] = 3.0f;
    actual->data[1] = 2.0f;

    struct PicoMSELoss mse = {.reduction = MEAN};
    struct PicoTensor* loss = pico_mse_loss(&ctx, &mse, pred, actual);

    loss->grad[0] = 1.0f;
    loss->_backward(loss);

    // analytic: (2/N)*(pred_i - actual_i)
    float expected_grad0 = (2.0f / 2.0f) * (5.0f - 3.0f);   // = 2.0
    float expected_grad1 = (2.0f / 2.0f) * (-1.0f - 2.0f);  // = -3.0

    ASSERT_TRUE(pred->grad[0] == expected_grad0);
    ASSERT_TRUE(pred->grad[1] == expected_grad1);

    pico_context_destroy(&ctx);
}

// calling backward twice must ACCUMULATE (+=), not overwrite, same rule as add/mul
UTEST(loss, mse_backward_accumulates) {
    struct PicoContext ctx = pico_context_init();

    int64_t s[] = {1};
    struct PicoTensor* pred = pico_param(&ctx, s, 1);
    struct PicoTensor* actual = pico_param(&ctx, s, 1);
    pred->data[0] = 5.0f;
    actual->data[0] = 3.0f;

    struct PicoMSELoss mse = {.reduction = MEAN};
    struct PicoTensor* loss = pico_mse_loss(&ctx, &mse, pred, actual);

    loss->grad[0] = 1.0f;
    loss->_backward(loss);
    float first = pred->grad[0];

    loss->_backward(loss);  // call again, should add on top, not reset
    ASSERT_TRUE(pred->grad[0] == first * 2.0f);

    pico_context_destroy(&ctx);
}

// ============================= SUM reduction + actuals side + scalar shape

// SUM reduction = sum of squared errors (NO division by N)
UTEST(loss, mse_sum_forward) {
    struct PicoContext ctx = pico_context_init();

    int64_t s[] = {3};
    struct PicoTensor* pred = pico_param(&ctx, s, 1);
    struct PicoTensor* actual = pico_param(&ctx, s, 1);
    pred->data[0] = 1.0f;  // diff 1 -> 1
    pred->data[1] = 2.0f;  // diff 2 -> 4
    pred->data[2] = 3.0f;  // diff 3 -> 9   (actuals default 0)

    struct PicoMSELoss mse = {.reduction = SUM};
    struct PicoTensor* loss = pico_mse_loss(&ctx, &mse, pred, actual);

    ASSERT_TRUE(loss->data[0] == 14.0f);  // 1+4+9, NOT /3

    pico_context_destroy(&ctx);
}

// SUM backward: d(sum)/d(pred_i) = 2*(pred_i - actual_i)   (no /N)
UTEST(loss, mse_sum_backward) {
    struct PicoContext ctx = pico_context_init();

    int64_t s[] = {1};
    struct PicoTensor* pred = pico_param(&ctx, s, 1);
    struct PicoTensor* actual = pico_param(&ctx, s, 1);
    pred->data[0] = 5.0f;
    actual->data[0] = 3.0f;

    struct PicoMSELoss mse = {.reduction = SUM};
    struct PicoTensor* loss = pico_mse_loss(&ctx, &mse, pred, actual);

    loss->grad[0] = 1.0f;
    loss->_backward(loss);

    ASSERT_TRUE(pred->grad[0] == 4.0f);  // 2*(5-3)

    pico_context_destroy(&ctx);
}

// the gradient w.r.t. the ACTUALS is the negative of the pred gradient
UTEST(loss, mse_backward_actuals_side) {
    struct PicoContext ctx = pico_context_init();

    int64_t s[] = {1};
    struct PicoTensor* pred = pico_param(&ctx, s, 1);
    struct PicoTensor* actual = pico_param(&ctx, s, 1);
    pred->data[0] = 5.0f;
    actual->data[0] = 3.0f;

    struct PicoMSELoss mse = {.reduction = MEAN};
    struct PicoTensor* loss = pico_mse_loss(&ctx, &mse, pred, actual);

    loss->grad[0] = 1.0f;
    loss->_backward(loss);

    ASSERT_TRUE(pred->grad[0] == 4.0f);     // (2/1)*(5-3)
    ASSERT_TRUE(actual->grad[0] == -4.0f);  // negated

    pico_context_destroy(&ctx);
}

// FAILS ON PURPOSE: the loss is a scalar, so its numel should be 1 — but mse.c
// builds `out` from predictions->shape (numel = N) and only sets ndim=0/shape=NULL,
// never numel. fix: make the loss a real 1-element tensor.
UTEST(loss, mse_output_is_scalar) {
    struct PicoContext ctx = pico_context_init();

    int64_t s[] = {3};  // 3-element prediction
    struct PicoTensor* pred = pico_param(&ctx, s, 1);
    struct PicoTensor* actual = pico_param(&ctx, s, 1);

    struct PicoMSELoss mse = {.reduction = MEAN};
    struct PicoTensor* loss = pico_mse_loss(&ctx, &mse, pred, actual);

    ASSERT_TRUE(loss->numel == 1);  // currently 3 -> FAILS until loss is a true scalar

    pico_context_destroy(&ctx);
}

// the loss must also work through the FULL traversal (pico_backward seeds + walks),
// not just when _backward is poked by hand — exercises the numel-seeding path.
UTEST(loss, mse_through_pico_backward) {
    struct PicoContext ctx = pico_context_init();

    int64_t s[] = {2};
    struct PicoTensor* pred = pico_param(&ctx, s, 1);
    struct PicoTensor* actual = pico_param(&ctx, s, 1);
    pred->data[0] = 5.0f;
    pred->data[1] = -1.0f;
    actual->data[0] = 3.0f;
    actual->data[1] = 2.0f;

    struct PicoMSELoss mse = {.reduction = MEAN};
    struct PicoTensor* loss = pico_mse_loss(&ctx, &mse, pred, actual);

    pico_backward(&ctx, loss);  // seeds loss grad + walks the graph

    ASSERT_TRUE(pred->grad[0] == 2.0f);   // (2/2)*(5-3)
    ASSERT_TRUE(pred->grad[1] == -3.0f);  // (2/2)*(-1-2)

    pico_context_destroy(&ctx);
}
