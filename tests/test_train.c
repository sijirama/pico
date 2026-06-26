/*
 * End-to-end "it trains" tests: forward -> loss -> pico_backward -> sgd_step,
 * then forward again and assert the loss went DOWN. The real proof the whole
 * engine (ops + autograd + loss + optimizer) works together.
 * NOTE: no UTEST_MAIN here, test_basic.c owns main + UTEST_STATE.
 * TODO(pico): manual optimizer cleanup until pico_optim_sgd_free exists.
 */
#include <stdlib.h>

#include "arena.h"
#include "lib/pico_vector.h"
#include "loss/loss.h"
#include "optim/optim.h"
#include "ops.h"
#include "tensor.h"
#include "utest.h"

// scalar regression: fit a single weight w toward a target. one SGD step must
// lower MSE.  w=0, target=5 -> loss=25 ; grad=2*(0-5)=-10 ; w -= 0.1*-10 = +1 ;
// new loss=(1-5)^2=16 < 25.
UTEST(train, step_lowers_loss_scalar) {
    struct Arena* ar = arena_init(8192);
    arena_ctx_push(ar);

    int64_t s[] = {1};
    struct PicoTensor* w = pico_param(s, 1);
    w->data[0] = 0.0f;
    struct PicoTensor* target = pico_param(s, 1);
    target->data[0] = 5.0f;

    struct PicoOptimSGD* opt = pico_optim_sgd_init(0.1f);
    pico_optim_sgd_add(opt, w);
    struct PicoMSELoss mse = {.reduction = MEAN};

    struct PicoTensor* loss1 = pico_mse_loss(&mse, w, target);
    float l1 = loss1->data[0];

    pico_optim_sgd_zero_grad(opt);
    pico_backward(ar, loss1);
    pico_optim_sgd_step(opt);

    struct PicoTensor* loss2 = pico_mse_loss(&mse, w, target);
    float l2 = loss2->data[0];

    ASSERT_TRUE(l2 < l1);  // it learned something

    pico_optim_sgd_free(opt);
    pico_free(w);
    pico_free(target);
    arena_ctx_pop();
    arena_destroy(ar);
}

// one-layer linear: pred = x @ w, fit toward target. full stack: matmul forward,
// matmul backward, mse backward, sgd step. one step must lower the loss.
UTEST(train, step_lowers_loss_linear) {
    struct Arena* ar = arena_init(8192);
    arena_ctx_push(ar);

    int64_t sx[] = {1, 2};
    struct PicoTensor* x = pico_param(sx, 2);
    x->data[0] = 1.0f;
    x->data[1] = 2.0f;

    int64_t sw[] = {2, 1};
    struct PicoTensor* w = pico_param(sw, 2);
    w->data[0] = 0.0f;
    w->data[1] = 0.0f;

    int64_t st[] = {1, 1};
    struct PicoTensor* target = pico_param(st, 2);
    target->data[0] = 5.0f;

    struct PicoOptimSGD* opt = pico_optim_sgd_init(0.01f);
    pico_optim_sgd_add(opt, w);  // only the weight is learnable
    struct PicoMSELoss mse = {.reduction = MEAN};

    struct PicoTensor* loss1 = pico_mse_loss(&mse, pico_matmul(x, w), target);
    float l1 = loss1->data[0];

    pico_optim_sgd_zero_grad(opt);
    pico_backward(ar, loss1);
    pico_optim_sgd_step(opt);

    struct PicoTensor* loss2 = pico_mse_loss(&mse, pico_matmul(x, w), target);
    float l2 = loss2->data[0];

    ASSERT_TRUE(l2 < l1);

    pico_optim_sgd_free(opt);
    pico_free(x);
    pico_free(w);
    pico_free(target);
    arena_ctx_pop();
    arena_destroy(ar);
}

// a few steps should keep driving the loss down (convergence-ish, scalar case)
UTEST(train, multi_step_keeps_improving) {
    struct Arena* ar = arena_init(1 << 16);
    arena_ctx_push(ar);

    int64_t s[] = {1};
    struct PicoTensor* w = pico_param(s, 1);
    w->data[0] = 0.0f;
    struct PicoTensor* target = pico_param(s, 1);
    target->data[0] = 5.0f;

    struct PicoOptimSGD* opt = pico_optim_sgd_init(0.1f);
    pico_optim_sgd_add(opt, w);
    struct PicoMSELoss mse = {.reduction = MEAN};

    float prev = 1e30f;
    for(int step = 0; step < 10; step++) {
        struct PicoTensor* loss = pico_mse_loss(&mse, w, target);
        float l = loss->data[0];
        ASSERT_TRUE(l < prev);  // strictly decreasing each step
        prev = l;

        pico_optim_sgd_zero_grad(opt);
        pico_backward(ar, loss);
        pico_optim_sgd_step(opt);
    }

    ASSERT_TRUE(prev < 1.0f);  // got close to the target

    pico_optim_sgd_free(opt);
    pico_free(w);
    pico_free(target);
    arena_ctx_pop();
    arena_destroy(ar);
}
