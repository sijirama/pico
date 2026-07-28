
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
    pico_init();
    struct PicoContext ctx = pico_context_init();

    int64_t shape[] = {3};
    float values[] = {1.0f, 2.0f, 3.0f};

    struct PicoTensor* x = pico_tensor_from_data(&ctx, shape, 1, values);
    struct PicoTensor* two = pico_tensor_from_scalar(&ctx, 2.0f);
    struct PicoTensor* y = pico_mul(&ctx, x, two);
    struct PicoTensor* z = pico_sqrt(&ctx, y);

    pico_tensor_print(z);
    // example output:
    // PicoTensor(shape=[3], numel=3)
    // [1.41421, 2, 2.44949]

    pico_context_destroy(&ctx);
    pico_shutdown();
    return 0;
}
```

LLM assistance is only used to for docs and tests
