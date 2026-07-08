#include <stdint.h>
#include <stdio.h>

#include "pico.h"

static void fill_dataset(struct PicoTensor* x, struct PicoTensor* y) {
    float xs[] = {
        0.0f, 0.0f,
        0.0f, 1.0f,
        1.0f, 0.0f,
        1.0f, 1.0f,
        2.0f, 0.0f,
        0.0f, 2.0f,
        2.0f, 1.0f,
        1.0f, 2.0f,
    };

    for(int i = 0; i < 16; i++) {
        x->data[i] = xs[i];
    }

    for(int row = 0; row < 8; row++) {
        float a = x->data[row * 2];
        float b = x->data[row * 2 + 1];
        y->data[row] = 1.0f + (2.0f * a) + (3.0f * b);
    }
}

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

static struct PicoTensor* forward(struct PicoLinear* l1, struct PicoLinear* l2,
                                  struct PicoTensor* x) {
    struct PicoTensor* h = pico_nn_linear_forward(l1, x);
    h = pico_relu(h);
    return pico_nn_linear_forward(l2, h);
}

int main(void) {
    pico_init();

    struct Arena* arena = arena_init(1 << 20);
    if(arena == NULL) {
        fprintf(stderr, "failed to create arena\n");
        return 1;
    }

    arena_ctx_push(arena);

    int64_t x_shape[] = {8, 2};
    int64_t y_shape[] = {8, 1};
    struct PicoTensor* x = pico_param(x_shape, 2);
    struct PicoTensor* y = pico_param(y_shape, 2);
    fill_dataset(x, y);

    struct PicoLinear* l1 = pico_nn_linear_init(2, 4, true);
    struct PicoLinear* l2 = pico_nn_linear_init(4, 1, true);
    init_layer_weights(l1, l2);

    struct PicoOptimSGD* opt = pico_optim_sgd_init(0.001f);
    pico_optim_sgd_add(opt, l1->weights);
    pico_optim_sgd_add(opt, l1->bias);
    pico_optim_sgd_add(opt, l2->weights);
    pico_optim_sgd_add(opt, l2->bias);

    struct PicoMSELoss mse = {.reduction = MEAN};

    printf("training: y = 1 + 2*x0 + 3*x1\n");
    printf("model: Linear(2,4,bias) -> ReLU -> Linear(4,1,bias)\n\n");

    for(int step = 0; step <= 800; step++) {
        struct PicoTensor* pred = forward(l1, l2, x);
        struct PicoTensor* loss = pico_mse_loss(&mse, pred, y);

        if(step % 100 == 0) {
            printf("step %3d | loss %.6f | pred[0] %.4f | target[0] %.4f\n", step,
                   loss->data[0], pred->data[0], y->data[0]);
        }

        pico_optim_sgd_zero_grad(opt);
        pico_backward(arena, loss);
        pico_optim_sgd_step(opt);

        arena_reset(arena);
    }

    struct PicoTensor* final_pred = forward(l1, l2, x);
    struct PicoTensor* final_loss = pico_mse_loss(&mse, final_pred, y);

    printf("\nfinal loss: %.6f\n", final_loss->data[0]);
    printf("\nfirst four predictions:\n");
    for(int i = 0; i < 4; i++) {
        printf("  row %d -> pred %.4f, target %.4f\n", i, final_pred->data[i], y->data[i]);
    }

    pico_optim_sgd_free(opt);
    pico_nn_linear_free(l1);
    pico_nn_linear_free(l2);
    pico_free(x);
    pico_free(y);
    arena_ctx_pop();
    arena_destroy(arena);

    return 0;
}
