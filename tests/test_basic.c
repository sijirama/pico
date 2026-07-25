#include "utest.h"
#include "pico.h"

UTEST(Basic, Version) {
    ASSERT_STREQ("0.01", PICO_VERSION);
}

UTEST(Basic, init_creates_default_arena_for_temp_ops) {
    pico_shutdown();
    ASSERT_TRUE(global_arena == NULL);
    ASSERT_TRUE(arena_ctx_current() == NULL);

    pico_init();
    ASSERT_TRUE(global_arena != NULL);
    ASSERT_TRUE(arena_ctx_current() == global_arena);

    int64_t shape[1] = {2};
    struct PicoTensor* a = pico_param(shape, 1);
    struct PicoTensor* b = pico_param(shape, 1);
    ASSERT_TRUE(a != NULL);
    ASSERT_TRUE(b != NULL);
    a->data[0] = 1.0f;
    a->data[1] = 2.0f;
    b->data[0] = 3.0f;
    b->data[1] = 4.0f;

    struct PicoTensor* out = pico_add(NULL, a, b);
    ASSERT_TRUE(out != NULL);
    ASSERT_NEAR(out->data[0], 4.0f, 1e-6f);
    ASSERT_NEAR(out->data[1], 6.0f, 1e-6f);

    pico_free(a);
    pico_free(b);
    pico_shutdown();
    ASSERT_TRUE(global_arena == NULL);
    ASSERT_TRUE(arena_ctx_current() == NULL);
}

UTEST_STATE();

int main(int argc, const char* const argv[]) {
    int result = utest_main(argc, argv);
    pico_shutdown();
    return result;
}
