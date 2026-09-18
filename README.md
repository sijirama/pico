
# pico

**a tiny machine-learning framework, written from scratch in C.**

---

The north star: **a tiny framework that's genuinely fast** — cache-aware kernels,
SIMD-vectorized math, arena-allocated graphs — not just correct.

## Tiny Example

```c
#include <stdint.h>
#include <stdio.h>

#include "pico.h"

int main(void) {
    struct PicoContext* ctx = pico_init();

    int64_t shape[] = {3};
    float values[] = {1.0f, 2.0f, 3.0f};

    struct PicoTensor* x = pico_tensor_from_data(ctx, shape, 1, values);
    struct PicoTensor* two = pico_tensor_from_scalar(ctx, 2.0f);
    struct PicoTensor* y = pico_mul(ctx, x, two);
    struct PicoTensor* z = pico_sqrt(ctx, y);

    pico_tensor_print(z);
    // example output:
    // PicoTensor(shape=[3], numel=3)
    // [1.41421, 2, 2.44949]

    pico_shutdown(ctx);
    return 0;
}
```

## Tiny Transformer Example

```c
#include <stdint.h>
#include <stdio.h>

#include "pico.h"
#include "safetensor/st.h"

int main(void) {
    struct PicoContext* ctx = pico_init();

    int64_t shape[] = {3, 4}; // [seq, embed]
    float x_values[] = {
        1, 0, 0, 0,
        0, 1, 0, 0,
        0, 0, 1, 0,
    };
    float y_values[] = {
        0.25f, 0, 0, 0,
        0, 0.25f, 0, 0,
        0, 0, 0.25f, 0,
    };

    struct PicoTensor* x = pico_tensor_from_data(ctx, shape, 2, x_values);
    struct PicoTensor* y = pico_tensor_from_data(ctx, shape, 2, y_values);

    struct PicoTransformer* block1 =
        pico_nn_transformer_init(ctx, "transformer.blocks.0", 4, 2, 2, 8);
    struct PicoTransformer* block2 =
        pico_nn_transformer_init(ctx, "transformer.blocks.1", 4, 2, 2, 8);

    struct PicoOptimSGD* opt = pico_optim_sgd_init(0.01f);
    struct PicoMSELoss mse = {.reduction = MEAN};

    for(int step = 0; step < 100; step++) {
        struct PicoTensor* h = pico_nn_transformer_forward(ctx, block1, x);
        struct PicoTensor* pred = pico_nn_transformer_forward(ctx, block2, h);
        struct PicoTensor* loss = pico_mse_loss(ctx, &mse, pred, y);

        pico_optim_sgd_zero_grad(ctx, opt);
        pico_backward(ctx, loss);
        pico_optim_sgd_step(ctx, opt);
    }

    save_tensor(ctx, "tiny_transformer.safetensors");

    pico_optim_sgd_free(opt);
    pico_nn_transformer_free(block1);
    pico_nn_transformer_free(block2);
    pico_shutdown(ctx);
    return 0;
}
```

There is a fuller runnable version in
`examples/07_train_transformer_save_safetensor`.

LLM assistance is mostly for docs, tests, and small implementation corners where
i don't really care about pretending i hand-rolled every tiny detail. `pico_permute`
and the RoPE implementation in self-attention are those kind of things.
