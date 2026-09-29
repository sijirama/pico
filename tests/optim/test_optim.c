/*
 * Tests for the optimizers.
 * NOTE: no UTEST_MAIN here, test_basic.c owns main + UTEST_STATE.
 * Values chosen to be exact in float (no 0.1-style rounding) so == is safe.
 */
#include <stdlib.h>

#include "ctx.h"
#include "global.h"
#include "optim/optim.h"
#include "tensor.h"
#include "utest.h"

// step does:  data -= lr * grad
UTEST(optim_sgd, step_updates_one_param) {
    struct PicoContext* ctx = pico_init_verbose(false);

    int64_t s[] = {1};
    struct PicoTensor* w = pico_param(ctx, s, 1);
    w->data[0] = 10.0f;
    w->grad[0] = 4.0f;

    struct PicoOptimSGD* opt = pico_optim_sgd_init(0.5f);
    pico_optim_sgd_step(ctx, opt);

    ASSERT_TRUE(w->data[0] == 8.0f);  // 10 - 0.5*4

    pico_optim_sgd_free(opt);
    pico_shutdown(ctx);
}

// step updates every element of a multi-element param
UTEST(optim_sgd, step_multi_element) {
    struct PicoContext* ctx = pico_init_verbose(false);

    int64_t s[] = {3};
    struct PicoTensor* w = pico_param(ctx, s, 1);
    float wd[] = {2, 4, 6};
    float wg[] = {2, 4, 6};
    for(int i = 0; i < 3; i++) {
        w->data[i] = wd[i];
        w->grad[i] = wg[i];
    }

    struct PicoOptimSGD* opt = pico_optim_sgd_init(0.5f);
    pico_optim_sgd_step(ctx, opt);

    ASSERT_TRUE(w->data[0] == 1.0f);  // 2 - 0.5*2
    ASSERT_TRUE(w->data[1] == 2.0f);  // 4 - 0.5*4
    ASSERT_TRUE(w->data[2] == 3.0f);  // 6 - 0.5*6

    pico_optim_sgd_free(opt);
    pico_shutdown(ctx);
}

// zero_grad clears all grads to 0
UTEST(optim_sgd, zero_grad_clears) {
    struct PicoContext* ctx = pico_init_verbose(false);

    int64_t s[] = {3};
    struct PicoTensor* w = pico_param(ctx, s, 1);
    for(int i = 0; i < 3; i++) w->grad[i] = 7.0f;

    struct PicoOptimSGD* opt = pico_optim_sgd_init(0.1f);
    pico_optim_sgd_zero_grad(ctx, opt);

    for(int i = 0; i < 3; i++) ASSERT_TRUE(w->grad[i] == 0.0f);

    pico_optim_sgd_free(opt);
    pico_shutdown(ctx);
}

// step updates ALL registered params, not just the first
UTEST(optim_sgd, step_multiple_params) {
    struct PicoContext* ctx = pico_init_verbose(false);

    int64_t s[] = {1};
    struct PicoTensor* w1 = pico_param(ctx, s, 1);
    struct PicoTensor* w2 = pico_param(ctx, s, 1);
    w1->data[0] = 10.0f;
    w1->grad[0] = 4.0f;
    w2->data[0] = 20.0f;
    w2->grad[0] = 10.0f;

    struct PicoOptimSGD* opt = pico_optim_sgd_init(0.5f);
    pico_optim_sgd_step(ctx, opt);

    ASSERT_TRUE(w1->data[0] == 8.0f);   // 10 - 0.5*4
    ASSERT_TRUE(w2->data[0] == 15.0f);  // 20 - 0.5*10

    pico_optim_sgd_free(opt);
    pico_shutdown(ctx);
}

UTEST(optim_adam, init_sets_default_hyperparams) {
    struct PicoOptimAdam* opt = pico_optim_adam_init(0.001f);

    ASSERT_TRUE(opt != NULL);
    ASSERT_NEAR(opt->lr, 0.001f, 1e-8f);
    ASSERT_NEAR(opt->beta1, 0.9f, 1e-6f);
    ASSERT_NEAR(opt->beta2, 0.999f, 1e-6f);
    ASSERT_NEAR(opt->eps, 1e-8f, 1e-10f);
    ASSERT_TRUE(opt->step == 0);

    pico_optim_adam_free(opt);
}

UTEST(optim_adam, first_step_uses_bias_corrected_moments) {
    struct PicoContext* ctx = pico_init_verbose(false);

    int64_t s[] = {2};
    struct PicoTensor* w = pico_param(ctx, s, 1);
    w->data[0] = 1.0f;
    w->data[1] = -1.0f;
    w->grad[0] = 0.5f;
    w->grad[1] = -0.25f;

    struct PicoOptimAdam* opt = pico_optim_adam_init(0.1f);
    pico_optim_adam_step(ctx, opt);

    ASSERT_TRUE(opt->step == 1);
    ASSERT_TRUE(opt->param_count == 1);
    ASSERT_TRUE(opt->params[0] == w);
    ASSERT_NEAR(opt->m[0][0], 0.05f, 1e-6f);
    ASSERT_NEAR(opt->v[0][0], 0.00025f, 1e-8f);
    ASSERT_NEAR(w->data[0], 0.9f, 1e-5f);
    ASSERT_NEAR(w->data[1], -0.9f, 1e-5f);

    pico_optim_adam_free(opt);
    pico_shutdown(ctx);
}

UTEST(optim_adam, second_step_reuses_state) {
    struct PicoContext* ctx = pico_init_verbose(false);

    int64_t s[] = {1};
    struct PicoTensor* w = pico_param(ctx, s, 1);
    w->data[0] = 1.0f;
    w->grad[0] = 1.0f;

    struct PicoOptimAdam* opt = pico_optim_adam_init(0.1f);
    pico_optim_adam_step(ctx, opt);
    w->grad[0] = 1.0f;
    pico_optim_adam_step(ctx, opt);

    ASSERT_TRUE(opt->step == 2);
    ASSERT_NEAR(opt->m[0][0], 0.19f, 1e-6f);
    ASSERT_NEAR(w->data[0], 0.8f, 1e-5f);

    pico_optim_adam_free(opt);
    pico_shutdown(ctx);
}

UTEST(optim_adam, zero_grad_clears_registered_params) {
    struct PicoContext* ctx = pico_init_verbose(false);

    int64_t s[] = {2};
    struct PicoTensor* w = pico_param(ctx, s, 1);
    w->grad[0] = 3.0f;
    w->grad[1] = -2.0f;

    struct PicoOptimAdam* opt = pico_optim_adam_init(0.1f);
    pico_optim_adam_zero_grad(ctx, opt);

    ASSERT_TRUE(w->grad[0] == 0.0f);
    ASSERT_TRUE(w->grad[1] == 0.0f);

    pico_optim_adam_free(opt);
    pico_shutdown(ctx);
}

UTEST(optim_adamw, init_sets_default_hyperparams_and_weight_decay) {
    struct PicoOptimAdamW* opt = pico_optim_adamw_init(0.001f, 0.01f);

    ASSERT_TRUE(opt != NULL);
    ASSERT_NEAR(opt->lr, 0.001f, 1e-8f);
    ASSERT_NEAR(opt->beta1, 0.9f, 1e-6f);
    ASSERT_NEAR(opt->beta2, 0.999f, 1e-6f);
    ASSERT_NEAR(opt->eps, 1e-8f, 1e-10f);
    ASSERT_NEAR(opt->weight_decay, 0.01f, 1e-8f);

    pico_optim_adamw_free(opt);
}

UTEST(optim_adamw, first_step_applies_decoupled_weight_decay) {
    struct PicoContext* ctx = pico_init_verbose(false);

    int64_t s[] = {1};
    struct PicoTensor* w = pico_param(ctx, s, 1);
    w->data[0] = 1.0f;
    w->grad[0] = 0.5f;

    struct PicoOptimAdamW* opt = pico_optim_adamw_init(0.1f, 0.01f);
    pico_optim_adamw_step(ctx, opt);

    ASSERT_TRUE(opt->step == 1);
    ASSERT_NEAR(w->data[0], 0.899f, 1e-5f);

    pico_optim_adamw_free(opt);
    pico_shutdown(ctx);
}

UTEST(optim_adamw, zero_grad_clears_registered_params) {
    struct PicoContext* ctx = pico_init_verbose(false);

    int64_t s[] = {2};
    struct PicoTensor* w = pico_param(ctx, s, 1);
    w->grad[0] = 3.0f;
    w->grad[1] = -2.0f;

    struct PicoOptimAdamW* opt = pico_optim_adamw_init(0.1f, 0.01f);
    pico_optim_adamw_zero_grad(ctx, opt);

    ASSERT_TRUE(w->grad[0] == 0.0f);
    ASSERT_TRUE(w->grad[1] == 0.0f);

    pico_optim_adamw_free(opt);
    pico_shutdown(ctx);
}
