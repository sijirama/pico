/*
 * Tests for the arena allocator (multi-block chained growth + ctx stack).
 */

#include <stdint.h>
#include <threads.h>

#include "arena.h"
#include "utest.h"

// helper: is ptr inside this block's backing buffer?
static int in_block(struct ArenaBlock* b, void* p) {
    uintptr_t lo = (uintptr_t)b->bottom;
    uintptr_t hi = lo + b->capacity;
    uintptr_t x = (uintptr_t)p;
    return x >= lo && x < hi;
}

// init should give us a live arena with one block ready to go
UTEST(arena, init_non_null) {
    struct Arena* a = arena_init(128);
    ASSERT_TRUE(a != NULL);
    ASSERT_TRUE(a->begin != NULL);
    ASSERT_TRUE(a->begin == a->end);  // one block at start
    arena_destroy(a);
}

// a basic alloc should hand back a usable, non-null pointer
UTEST(arena, alloc_non_null) {
    struct Arena* a = arena_init(128);
    void* p = arena_alloc(a, 16);
    ASSERT_TRUE(p != NULL);
    arena_destroy(a);
}

// the memory we get back should actually be writable and readable
UTEST(arena, alloc_is_writable) {
    struct Arena* a = arena_init(128);
    int* p = (int*)arena_alloc(a, sizeof(int) * 4);
    for(int i = 0; i < 4; i++)
        p[i] = i * 10;
    ASSERT_EQ(p[0], 0);
    ASSERT_EQ(p[3], 30);
    arena_destroy(a);
}

// two separate allocs must not overlap, second sits right after the first (bump)
UTEST(arena, allocs_bump_forward) {
    struct Arena* a = arena_init(128);
    unsigned char* p0 = (unsigned char*)arena_alloc(a, 16);
    unsigned char* p1 = (unsigned char*)arena_alloc(a, 16);
    ASSERT_TRUE(p1 == p0 + 16);  // pure bump, no gap
    arena_destroy(a);
}

// when the first block fills up, the arena should grow a new block
UTEST(arena, grows_when_full) {
    struct Arena* a = arena_init(64);
    arena_alloc(a, 64);  // fills block 1 exactly
    arena_alloc(a, 8);   // should trigger a new block
    ASSERT_TRUE(a->begin->next != NULL);
    ASSERT_TRUE(a->begin != a->end);
    arena_destroy(a);
}

// the key arena property: old pointers stay valid after the arena grows
UTEST(arena, data_survives_growth) {
    struct Arena* a = arena_init(64);
    int* first = (int*)arena_alloc(a, sizeof(int));
    *first = 1234;

    int* second = (int*)arena_alloc(a, 64);  // forces growth into a new block
    *second = 5678;

    ASSERT_TRUE(in_block(a->begin, first));  // first lives in block 1
    ASSERT_EQ(*first, 1234);                 // ...and is untouched by the growth
    ASSERT_EQ(*second, 5678);
    arena_destroy(a);
}

// after reset, a fresh alloc should reuse the very start of the block
UTEST(arena, reset_reuses_memory) {
    struct Arena* a = arena_init(128);
    void* p0 = arena_alloc(a, 16);
    arena_alloc(a, 16);
    arena_reset(a);
    void* p1 = arena_alloc(a, 16);
    ASSERT_TRUE(p1 == p0);  // back to the top of the block
    arena_destroy(a);
}

// after growth then reset, allocation should come back to the FIRST block
UTEST(arena, reset_after_growth) {
    struct Arena* a = arena_init(64);
    void* p0 = arena_alloc(a, 8);  // in block 1
    arena_alloc(a, 64);            // grow to block 2
    arena_reset(a);
    void* p1 = arena_alloc(a, 8);  // should be back at the start of block 1
    ASSERT_TRUE(in_block(a->begin, p1));
    ASSERT_TRUE(p1 == p0);
    arena_destroy(a);
}

// ============================ arena context stack

// with nothing pushed, current should be null
UTEST(arena_ctx, current_empty_is_null) {
    ASSERT_TRUE(arena_ctx_current() == NULL);
}

// after pushing an arena it should be the current one
UTEST(arena_ctx, push_then_current) {
    struct Arena* a = arena_init(64);
    arena_ctx_push(a);
    ASSERT_TRUE(arena_ctx_current() == a);
    arena_ctx_pop();
    arena_destroy(a);
}

// pushing two then popping should restore the previous one (lifo)
UTEST(arena_ctx, pop_restores_previous) {
    struct Arena* a = arena_init(64);
    struct Arena* b = arena_init(64);
    arena_ctx_push(a);
    arena_ctx_push(b);
    ASSERT_TRUE(arena_ctx_current() == b);
    arena_ctx_pop();
    ASSERT_TRUE(arena_ctx_current() == a);
    arena_ctx_pop();
    ASSERT_TRUE(arena_ctx_current() == NULL);
    arena_destroy(a);
    arena_destroy(b);
}

// ============================ multithreaded ctx (thread_local stack)
//
// the ctx stack is thread_local, so every thread has its OWN stack. each thread
// pushes its own arena and must ALWAYS see that same arena as current, no matter
// what the other threads are doing. if the stack were shared/global, threads
// would clobber each other and somebody would see the wrong arena (or null).
// ASSERT_* can't run inside a thread fn, so each thread reports ok via its arg.

#define ARENA_TEST_NTHREADS 8
#define ARENA_TEST_SPINS 2000

struct arena_thread_arg {
    int ok;
};

static int arena_thread_body(void* p) {
    struct arena_thread_arg* arg = (struct arena_thread_arg*)p;
    arg->ok = 0;

    struct Arena* mine = arena_init(64);
    if(mine == NULL)
        return 0;

    // this thread's own stack should start empty
    if(arena_ctx_current() != NULL) {
        arena_destroy(mine);
        return 0;
    }

    arena_ctx_push(mine);

    // spin so the threads interleave; current must stay == our own arena
    for(int i = 0; i < ARENA_TEST_SPINS; i++) {
        if(arena_ctx_current() != mine) {
            arena_ctx_pop();
            arena_destroy(mine);
            return 0;
        }
        thrd_yield();  // encourage interleaving to surface any sharing bug
    }

    arena_ctx_pop();
    if(arena_ctx_current() != NULL) {
        arena_destroy(mine);
        return 0;
    }

    arena_destroy(mine);
    arg->ok = 1;
    return 0;
}

// each thread should see only its own arena on its own ctx stack
UTEST(arena_ctx, threads_have_independent_stacks) {
    thrd_t threads[ARENA_TEST_NTHREADS];
    struct arena_thread_arg args[ARENA_TEST_NTHREADS];

    for(int i = 0; i < ARENA_TEST_NTHREADS; i++) {
        args[i].ok = 0;
        ASSERT_EQ(thrd_create(&threads[i], arena_thread_body, &args[i]), thrd_success);
    }
    for(int i = 0; i < ARENA_TEST_NTHREADS; i++) {
        thrd_join(threads[i], NULL);
    }
    for(int i = 0; i < ARENA_TEST_NTHREADS; i++) {
        ASSERT_EQ(args[i].ok, 1);  // every thread saw only its own arena
    }
}

// the main thread's stack is unaffected by what worker threads pushed
UTEST(arena_ctx, main_stack_isolated_from_workers) {
    struct Arena* mainArena = arena_init(64);
    arena_ctx_push(mainArena);

    thrd_t threads[ARENA_TEST_NTHREADS];
    struct arena_thread_arg args[ARENA_TEST_NTHREADS];
    for(int i = 0; i < ARENA_TEST_NTHREADS; i++) {
        args[i].ok = 0;
        ASSERT_EQ(thrd_create(&threads[i], arena_thread_body, &args[i]), thrd_success);
    }
    for(int i = 0; i < ARENA_TEST_NTHREADS; i++) {
        thrd_join(threads[i], NULL);
    }

    // after all the worker churn, main still sees its own arena
    ASSERT_TRUE(arena_ctx_current() == mainArena);

    arena_ctx_pop();
    arena_destroy(mainArena);
}
