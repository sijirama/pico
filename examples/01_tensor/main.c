#include <stdint.h>
#include <stdio.h>

#include "pico.h"

int main(void) {
    pico_init();

    struct Arena* arena = arena_init(1024 * 1024);
    if(arena == NULL) {
        fprintf(stderr, "failed to create arena\n");
        return 1;
    }

    arena_ctx_push(arena);

    int64_t shape[1] = {3};
    struct PicoTensor* x = pico_rand(arena, shape, 1);
    struct PicoTensor* scale = pico_tensor_from_scalar(2.0f);
    struct PicoTensor* y = pico_mul(x, scale);

    printf("x: ");
    for(int64_t i = 0; i < x->numel; i++) {
        printf("%f ", x->data[i]);
    }
    printf("\n");

    printf("y = x * 2: ");
    for(int64_t i = 0; i < y->numel; i++) {
        printf("%f ", y->data[i]);
    }
    printf("\n");

    arena_ctx_pop();
    arena_destroy(arena);

    return 0;
}
