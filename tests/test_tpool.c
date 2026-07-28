#include "utest.h"

#include <stdatomic.h>

#include "global.h"
#include "tpool.h"

struct tpool_counter_arg {
    atomic_int* counter;
};

static void tpool_increment(void* p) {
    struct tpool_counter_arg* arg = (struct tpool_counter_arg*)p;
    atomic_fetch_add(arg->counter, 1);
}

UTEST(tpool, create_and_destroy) {
    struct PicoTPool* tp = pico_tpool_create(2);
    ASSERT_TRUE(tp != NULL);
    pico_tpool_destroy(tp);
}

UTEST(tpool, zero_threads_uses_default) {
    struct PicoTPool* tp = pico_tpool_create(0);
    ASSERT_TRUE(tp != NULL);
    ASSERT_EQ(tp->thread_cnt, 2);
    pico_tpool_destroy(tp);
}

UTEST(tpool, wait_empty_pool_returns) {
    struct PicoTPool* tp = pico_tpool_create(2);
    ASSERT_TRUE(tp != NULL);
    pico_tpool_wait(tp);
    pico_tpool_destroy(tp);
}

UTEST(tpool, runs_all_enqueued_jobs) {
    struct PicoTPool* tp = pico_tpool_create(4);
    ASSERT_TRUE(tp != NULL);

    atomic_int counter;
    atomic_init(&counter, 0);
    struct tpool_counter_arg arg = {.counter = &counter};

    for(int i = 0; i < 128; i++) {
        ASSERT_TRUE(pico_tpool_add_work(tp, tpool_increment, &arg));
    }

    pico_tpool_wait(tp);
    ASSERT_EQ(atomic_load(&counter), 128);

    pico_tpool_destroy(tp);
}

UTEST(tpool, rejects_null_pool) {
    atomic_int counter;
    atomic_init(&counter, 0);
    struct tpool_counter_arg arg = {.counter = &counter};

    ASSERT_FALSE(pico_tpool_add_work(NULL, tpool_increment, &arg));
    ASSERT_EQ(atomic_load(&counter), 0);
}

UTEST(tpool, rejects_null_work_function) {
    struct PicoTPool* tp = pico_tpool_create(2);
    ASSERT_TRUE(tp != NULL);

    ASSERT_FALSE(pico_tpool_add_work(tp, NULL, NULL));

    pico_tpool_destroy(tp);
}

UTEST(tpool, global_init_shutdown_lifecycle) {
    pico_shutdown();
    ASSERT_TRUE(global_tp == NULL);
    ASSERT_EQ(g_pico_initialized, 0);

    pico_init();
    ASSERT_TRUE(global_tp != NULL);
    ASSERT_EQ(g_pico_initialized, 1);

    pico_shutdown();
    ASSERT_TRUE(global_tp == NULL);
    ASSERT_EQ(g_pico_initialized, 0);
}
