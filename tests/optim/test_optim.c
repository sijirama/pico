/*
 * Tests for the SGD optimizer.
 * NOTE: no UTEST_MAIN here, test_basic.c owns main + UTEST_STATE.
 * Values chosen to be exact in float (no 0.1-style rounding) so == is safe.
 */
#include <stdlib.h>

#include "ctx.h"
#include "optim/optim.h"
#include "tensor.h"
#include "utest.h"

// step does:  data -= lr * grad
UTEST(optim_sgd, step_updates_one_param) {
    struct PicoContext ctx = pico_context_init();

    int64_t s[] = {1};
    struct PicoTensor* w = pico_param(&ctx, s, 1);
    w->data[0] = 10.0f;
    w->grad[0] = 4.0f;

    struct PicoOptimSGD* opt = pico_optim_sgd_init(0.5f);
    pico_optim_sgd_step(&ctx, opt);

    ASSERT_TRUE(w->data[0] == 8.0f);  // 10 - 0.5*4

    pico_optim_sgd_free(opt);
    pico_context_destroy(&ctx);
}

// step updates every element of a multi-element param
UTEST(optim_sgd, step_multi_element) {
    struct PicoContext ctx = pico_context_init();

    int64_t s[] = {3};
    struct PicoTensor* w = pico_param(&ctx, s, 1);
    float wd[] = {2, 4, 6};
    float wg[] = {2, 4, 6};
    for(int i = 0; i < 3; i++) {
        w->data[i] = wd[i];
        w->grad[i] = wg[i];
    }

    struct PicoOptimSGD* opt = pico_optim_sgd_init(0.5f);
    pico_optim_sgd_step(&ctx, opt);

    ASSERT_TRUE(w->data[0] == 1.0f);  // 2 - 0.5*2
    ASSERT_TRUE(w->data[1] == 2.0f);  // 4 - 0.5*4
    ASSERT_TRUE(w->data[2] == 3.0f);  // 6 - 0.5*6

    pico_optim_sgd_free(opt);
    pico_context_destroy(&ctx);
}

// zero_grad clears all grads to 0
UTEST(optim_sgd, zero_grad_clears) {
    struct PicoContext ctx = pico_context_init();

    int64_t s[] = {3};
    struct PicoTensor* w = pico_param(&ctx, s, 1);
    for(int i = 0; i < 3; i++) w->grad[i] = 7.0f;

    struct PicoOptimSGD* opt = pico_optim_sgd_init(0.1f);
    pico_optim_sgd_zero_grad(&ctx, opt);

    for(int i = 0; i < 3; i++) ASSERT_TRUE(w->grad[i] == 0.0f);

    pico_optim_sgd_free(opt);
    pico_context_destroy(&ctx);
}

// step updates ALL registered params, not just the first
UTEST(optim_sgd, step_multiple_params) {
    struct PicoContext ctx = pico_context_init();

    int64_t s[] = {1};
    struct PicoTensor* w1 = pico_param(&ctx, s, 1);
    struct PicoTensor* w2 = pico_param(&ctx, s, 1);
    w1->data[0] = 10.0f;
    w1->grad[0] = 4.0f;
    w2->data[0] = 20.0f;
    w2->grad[0] = 10.0f;

    struct PicoOptimSGD* opt = pico_optim_sgd_init(0.5f);
    pico_optim_sgd_step(&ctx, opt);

    ASSERT_TRUE(w1->data[0] == 8.0f);   // 10 - 0.5*4
    ASSERT_TRUE(w2->data[0] == 15.0f);  // 20 - 0.5*10

    pico_optim_sgd_free(opt);
    pico_context_destroy(&ctx);
}
