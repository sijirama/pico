#include <stdint.h>
#include <stdio.h>

#include "pico.h"

static void init_layer_weights(struct PicoLinear* l1, struct PicoLinear* l2) {
    float w1[] = {
        0.50f, 0.10f, 0.20f, 0.30f,
        0.10f, 0.60f, 0.40f, 0.20f,
    };

    float w2[] = {
        0.20f,
        0.10f,
        0.30f,
        0.20f,
    };

    for(int i = 0; i < l1->weights->numel; i++) {
        l1->weights->data[i] = w1[i];
    }

    for(int i = 0; i < l2->weights->numel; i++) {
        l2->weights->data[i] = w2[i];
    }

    for(int i = 0; i < l1->bias->numel; i++) {
        l1->bias->data[i] = 0.10f;
    }

    l2->bias->data[0] = 0.10f;
}

static struct PicoTensor* forward(struct PicoContext* ctx, struct PicoLinear* l1, struct PicoLinear* l2,
                                  struct PicoTensor* x) {
    struct PicoTensor* h = pico_nn_linear_forward(ctx, l1, x);
    h = pico_relu(ctx, h);
    return pico_nn_linear_forward(ctx, l2, h);
}

int main(void) {
    pico_init();

    struct PicoContext data_ctx = pico_context_init();
    struct PicoContext train_ctx = pico_context_init();
    if(data_ctx.arena == NULL || train_ctx.arena == NULL) {
        fprintf(stderr, "failed to create context\n");
        return 1;
    }

    int64_t x_shape[] = {8, 2};
    int64_t y_shape[] = {8, 1};
    float x_values[] = {
        0.0f, 0.0f,
        0.0f, 1.0f,
        1.0f, 0.0f,
        1.0f, 1.0f,
        2.0f, 0.0f,
        0.0f, 2.0f,
        2.0f, 1.0f,
        1.0f, 2.0f,
    };
    float y_values[] = {
        1.0f,
        4.0f,
        3.0f,
        6.0f,
        5.0f,
        7.0f,
        8.0f,
        9.0f,
    };

    struct PicoTensor* x = pico_tensor_from_data(&data_ctx, x_shape, 2, x_values);
    struct PicoTensor* y = pico_tensor_from_data(&data_ctx, y_shape, 2, y_values);

    struct PicoLinear* l1 = pico_nn_linear_init(&train_ctx, 2, 4, true);
    struct PicoLinear* l2 = pico_nn_linear_init(&train_ctx, 4, 1, true);
    init_layer_weights(l1, l2);

    struct PicoOptimSGD* opt = pico_optim_sgd_init(0.001f);

    struct PicoMSELoss mse = {.reduction = MEAN};

    printf("training: y = 1 + 2*x0 + 3*x1\n");
    printf("model: Linear(2,4,bias) -> ReLU -> Linear(4,1,bias)\n\n");

    for(int step = 0; step <= 800; step++) {
        struct PicoTensor* pred = forward(&train_ctx, l1, l2, x);
        struct PicoTensor* loss = pico_mse_loss(&train_ctx, &mse, pred, y);

        if(step % 100 == 0) {
            printf("step %3d | loss %.6f | pred[0] %.4f | target[0] %.4f\n", step,
                   loss->data[0], pred->data[0], y->data[0]);
        }

        pico_optim_sgd_zero_grad(&train_ctx, opt);
        pico_backward(&train_ctx, loss);
        pico_optim_sgd_step(&train_ctx, opt);

        arena_reset(train_ctx.arena);
    }

    struct PicoTensor* final_pred = forward(&train_ctx, l1, l2, x);
    struct PicoTensor* final_loss = pico_mse_loss(&train_ctx, &mse, final_pred, y);

    printf("\nfinal loss: %.6f\n", final_loss->data[0]);
    printf("\nfirst four predictions:\n");
    for(int i = 0; i < 4; i++) {
        printf("  row %d -> pred %.4f, target %.4f\n", i, final_pred->data[i], y->data[i]);
    }

    pico_optim_sgd_free(opt);
    pico_nn_linear_free(l1);
    pico_nn_linear_free(l2);
    pico_context_destroy(&train_ctx);
    pico_context_destroy(&data_ctx);
    pico_shutdown();

    return 0;
}
