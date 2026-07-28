#include "utest.h"
#include "pico.h"

UTEST(Basic, Version) {
    ASSERT_STREQ("0.01", PICO_VERSION);
}

UTEST(Basic, init_shutdown_and_explicit_context_ops) {
    pico_shutdown();

    pico_init();
    ASSERT_TRUE(g_pico_initialized);

    struct PicoContext ctx = pico_context_init();

    int64_t shape[1] = {2};
    struct PicoTensor* a = pico_param(&ctx, shape, 1);
    struct PicoTensor* b = pico_param(&ctx, shape, 1);
    ASSERT_TRUE(a != NULL);
    ASSERT_TRUE(b != NULL);
    a->data[0] = 1.0f;
    a->data[1] = 2.0f;
    b->data[0] = 3.0f;
    b->data[1] = 4.0f;

    struct PicoTensor* out = pico_add(&ctx, a, b);
    ASSERT_TRUE(out != NULL);
    ASSERT_NEAR(out->data[0], 4.0f, 1e-6f);
    ASSERT_NEAR(out->data[1], 6.0f, 1e-6f);

    pico_context_destroy(&ctx);

    pico_shutdown();
    ASSERT_FALSE(g_pico_initialized);
}

UTEST_STATE();

int main(int argc, const char* const argv[]) {
    int result = utest_main(argc, argv);
    pico_shutdown();
    return result;
}
