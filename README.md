
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

    int64_t shape[] = {3};
    float values[] = {1.0f, 2.0f, 3.0f};

    struct PicoTensor* x = pico_tensor_from_data(NULL, shape, 1, values);
    struct PicoTensor* two = pico_tensor_from_scalar(NULL, 2.0f);
    struct PicoTensor* y = pico_mul(NULL, x, two);
    struct PicoTensor* z = pico_sqrt(NULL, y);

    pico_tensor_print(z);

    pico_shutdown();
    return 0;
}
```

LLM assistance is only used to for docs and tests
