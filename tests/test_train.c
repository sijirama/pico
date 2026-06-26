/*
 * End-to-end "it trains" tests: forward -> loss -> pico_backward -> sgd_step,
 * then forward again and assert the loss went DOWN. The real proof the whole
 * engine (ops + autograd + loss + optimizer) works together.
 * NOTE: no UTEST_MAIN here, test_basic.c owns main + UTEST_STATE.
 * TODO(pico): manual optimizer cleanup until pico_optim_sgd_free exists.
 */
#include <stdlib.h>

#include "act/activations.h"
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

// HEADLINE: a 2-layer MLP with a relu in the middle.
//   pred = relu(x @ W1) @ W2 ;  loss = mse(pred, target)
// proves the gradient flows back THROUGH the relu gate (matmul -> relu -> matmul)
// and the whole net trains. W1 starts as identity so the relu gates open and stay
// open; W2 starts at 0 so step 1 learns via W2, later steps reach W1 too.
UTEST(train, relu_mlp_lowers_loss) {
    struct Arena* ar = arena_init(1 << 16);
    arena_ctx_push(ar);

    int64_t sx[] = {1, 2};
    struct PicoTensor* x = pico_param(sx, 2);
    x->data[0] = 1.0f;
    x->data[1] = 1.0f;

    int64_t s1[] = {2, 2};
    struct PicoTensor* W1 = pico_param(s1, 2);
    W1->data[0] = 1.0f;  // identity
    W1->data[1] = 0.0f;
    W1->data[2] = 0.0f;
    W1->data[3] = 1.0f;

    int64_t s2[] = {2, 1};
    struct PicoTensor* W2 = pico_param(s2, 2);
    W2->data[0] = 0.0f;
    W2->data[1] = 0.0f;

    int64_t st[] = {1, 1};
    struct PicoTensor* target = pico_param(st, 2);
    target->data[0] = 1.0f;

    struct PicoOptimSGD* opt = pico_optim_sgd_init(0.1f);
    pico_optim_sgd_add(opt, W1);
    pico_optim_sgd_add(opt, W2);
    struct PicoMSELoss mse = {.reduction = MEAN};

    float prev = 1e30f;
    for(int step = 0; step < 5; step++) {
        struct PicoTensor* pred = pico_matmul(pico_relu(pico_matmul(x, W1)), W2);
        struct PicoTensor* loss = pico_mse_loss(&mse, pred, target);

        float l = loss->data[0];
        ASSERT_TRUE(l < prev);  // strictly decreasing every step
        prev = l;

        pico_optim_sgd_zero_grad(opt);
        pico_backward(ar, loss);
        pico_optim_sgd_step(opt);
    }

    ASSERT_TRUE(prev < 1.0f);  // started at 1.0, learned

    pico_optim_sgd_free(opt);
    pico_free(x);
    pico_free(W1);
    pico_free(W2);
    pico_free(target);
    arena_ctx_pop();
    arena_destroy(ar);
}
