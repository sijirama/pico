/*
 * Tests for pico_mse_loss (forward value, graph wiring, backward correctness).
 * NOTE: no UTEST_MAIN here, test_basic.c owns main + UTEST_STATE.
 */

#include "arena.h"
#include "loss/loss.h"
#include "ops.h"
#include "tensor.h"
#include "utest.h"

// MSE of a single element: (pred - actual)^2 / 1
UTEST(loss, mse_forward_single_element) {
    struct Arena* ar = arena_init(4096);
    arena_ctx_push(ar);

    int64_t s[] = {1};
    struct PicoTensor* pred = pico_param(s, 1);
    struct PicoTensor* actual = pico_param(s, 1);
    pred->data[0] = 5.0f;
    actual->data[0] = 3.0f;

    struct PicoMSELoss mse = {.reduction = MEAN};
    struct PicoTensor* loss = pico_mse_loss(&mse, pred, actual);

    // (5-3)^2 = 4
    ASSERT_TRUE(loss->data[0] == 4.0f);

    pico_free(pred);
    pico_free(actual);
    arena_ctx_pop();
    arena_destroy(ar);
}

// MSE across multiple elements should be the MEAN of squared errors, not the sum
UTEST(loss, mse_forward_multi_element_mean) {
    struct Arena* ar = arena_init(4096);
    arena_ctx_push(ar);

    int64_t s[] = {3};
    struct PicoTensor* pred = pico_param(s, 1);
    struct PicoTensor* actual = pico_param(s, 1);

    pred->data[0] = 1.0f;
    actual->data[0] = 0.0f;  // diff 1 -> sq 1
    pred->data[1] = 2.0f;
    actual->data[1] = 0.0f;  // diff 2 -> sq 4
    pred->data[2] = 3.0f;
    actual->data[2] = 0.0f;  // diff 3 -> sq 9

    struct PicoMSELoss mse = {.reduction = MEAN};
    struct PicoTensor* loss = pico_mse_loss(&mse, pred, actual);

    // (1+4+9)/3 = 4.666...
    ASSERT_TRUE(loss->data[0] > 4.66f && loss->data[0] < 4.67f);

    pico_free(pred);
    pico_free(actual);
    arena_ctx_pop();
    arena_destroy(ar);
}

// pico_mse_loss should wire the graph: two parents (pred, actual) + backward fn attached
UTEST(loss, mse_wires_graph) {
    struct Arena* ar = arena_init(4096);
    arena_ctx_push(ar);

    int64_t s[] = {1};
    struct PicoTensor* pred = pico_param(s, 1);
    struct PicoTensor* actual = pico_param(s, 1);

    struct PicoMSELoss mse = {.reduction = MEAN};
    struct PicoTensor* loss = pico_mse_loss(&mse, pred, actual);

    ASSERT_EQ(loss->num_parents, 2);
    ASSERT_TRUE(loss->parents[0] == pred);
    ASSERT_TRUE(loss->parents[1] == actual);
    ASSERT_TRUE(loss->_backward != NULL);

    pico_free(pred);
    pico_free(actual);
    arena_ctx_pop();
    arena_destroy(ar);
}

// d(MSE)/d(pred_i) = (2/N) * (pred_i - actual_i), verified against numeric finite difference
UTEST(loss, mse_backward_matches_finite_difference) {
    struct Arena* ar = arena_init(4096);
    arena_ctx_push(ar);

    int64_t s[] = {2};
    struct PicoTensor* pred = pico_param(s, 1);
    struct PicoTensor* actual = pico_param(s, 1);
    pred->data[0] = 5.0f;
    pred->data[1] = -1.0f;
    actual->data[0] = 3.0f;
    actual->data[1] = 2.0f;

    struct PicoMSELoss mse = {.reduction = MEAN};
    struct PicoTensor* loss = pico_mse_loss(&mse, pred, actual);

    loss->grad[0] = 1.0f;
    loss->_backward(loss);

    // analytic: (2/N)*(pred_i - actual_i)
    float expected_grad0 = (2.0f / 2.0f) * (5.0f - 3.0f);   // = 2.0
    float expected_grad1 = (2.0f / 2.0f) * (-1.0f - 2.0f);  // = -3.0

    ASSERT_TRUE(pred->grad[0] == expected_grad0);
    ASSERT_TRUE(pred->grad[1] == expected_grad1);

    pico_free(pred);
    pico_free(actual);
    arena_ctx_pop();
    arena_destroy(ar);
}

// calling backward twice must ACCUMULATE (+=), not overwrite, same rule as add/mul
UTEST(loss, mse_backward_accumulates) {
    struct Arena* ar = arena_init(4096);
    arena_ctx_push(ar);

    int64_t s[] = {1};
    struct PicoTensor* pred = pico_param(s, 1);
    struct PicoTensor* actual = pico_param(s, 1);
    pred->data[0] = 5.0f;
    actual->data[0] = 3.0f;

    struct PicoMSELoss mse = {.reduction = MEAN};
    struct PicoTensor* loss = pico_mse_loss(&mse, pred, actual);

    loss->grad[0] = 1.0f;
    loss->_backward(loss);
    float first = pred->grad[0];

    loss->_backward(loss);  // call again, should add on top, not reset
    ASSERT_TRUE(pred->grad[0] == first * 2.0f);

    pico_free(pred);
    pico_free(actual);
    arena_ctx_pop();
    arena_destroy(ar);
}
