#include <stdint.h>
#include <stdio.h>

#include "pico.h"

int main(void) {
    pico_init();
    struct PicoContext ctx = pico_context_init();

    int64_t shape[1] = {3};
    float values[] = {1.0f, 2.0f, 3.0f};

    struct PicoTensor* x = pico_tensor_from_data(&ctx, shape, 1, values);
    struct PicoTensor* scale = pico_tensor_from_scalar(&ctx, 2.0f);
    struct PicoTensor* y = pico_mul(&ctx, x, scale);
    struct PicoTensor* z = pico_sqrt(&ctx, y);

    printf("x:\n");
    pico_tensor_print(x);

    printf("y = x * 2:\n");
    pico_tensor_print(y);

    printf("sqrt(y):\n");
    pico_tensor_print(z);

    pico_context_destroy(&ctx);
    pico_shutdown();
    return 0;
}
