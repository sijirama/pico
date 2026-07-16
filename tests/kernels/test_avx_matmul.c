/*
 * Tests for the AVX matmul kernel (pico_matmul_cpu_avx).
 * Forces g_simd_level = SIMD_AVX so pico_matmul routes to the AVX kernel, then
 * restores it (save/restore, guarded by __builtin_cpu_supports). Asserts the
 * mathematically correct C = A@B — a mix incl. edge cases. WIP kernel: may fail.
 * NOTE: no UTEST_MAIN here, test_basic.c owns main + UTEST_STATE.
 */
#include "arena.h"
#include "global.h"
#include "ops.h"
#include "tensor.h"
#include "utest.h"

// basic 2x2 @ 2x2. [[1,2],[3,4]] @ [[5,6],[7,8]] = [[19,22],[43,50]]
UTEST(avx_matmul, square_2x2) {
    if(!__builtin_cpu_supports("avx"))
        return;
    SimdLevel saved = g_simd_level;
    pico_init();
    g_simd_level = SIMD_AVX;

    struct Arena* ar = arena_init(1 << 16);
    arena_ctx_push(ar);

    int64_t sa[] = {2, 2};
    int64_t sb[] = {2, 2};
    struct PicoTensor* a = pico_param(sa, 2);
    struct PicoTensor* b = pico_param(sb, 2);
    float av[] = {1, 2, 3, 4};
    float bv[] = {5, 6, 7, 8};
    for(int i = 0; i < 4; i++) {
        a->data[i] = av[i];
        b->data[i] = bv[i];
    }

    struct PicoTensor* out = pico_matmul(a, b);
    pico_tensor_print(out);  // see what the AVX matmul actually produced
    float o0 = out->data[0], o1 = out->data[1], o2 = out->data[2], o3 = out->data[3];

    pico_free(a);
    pico_free(b);
    arena_ctx_pop();
    arena_destroy(ar);
    g_simd_level = saved;

    ASSERT_TRUE(o0 == 19.0f);
    ASSERT_TRUE(o1 == 22.0f);
    ASSERT_TRUE(o2 == 43.0f);
    ASSERT_TRUE(o3 == 50.0f);
}

// non-square 2x3 @ 3x2 = 2x2. [[58,64],[139,154]]
UTEST(avx_matmul, nonsquare_2x3_3x2) {
    if(!__builtin_cpu_supports("avx"))
        return;
    SimdLevel saved = g_simd_level;
    pico_init();
    g_simd_level = SIMD_AVX;

    struct Arena* ar = arena_init(1 << 16);
    arena_ctx_push(ar);

    int64_t sa[] = {2, 3};
    int64_t sb[] = {3, 2};
    struct PicoTensor* a = pico_param(sa, 2);
    struct PicoTensor* b = pico_param(sb, 2);
    float av[] = {1, 2, 3, 4, 5, 6};
    float bv[] = {7, 8, 9, 10, 11, 12};
    for(int i = 0; i < 6; i++) {
        a->data[i] = av[i];
        b->data[i] = bv[i];
    }

    struct PicoTensor* out = pico_matmul(a, b);
    float o0 = out->data[0], o1 = out->data[1], o2 = out->data[2], o3 = out->data[3];

    pico_free(a);
    pico_free(b);
    arena_ctx_pop();
    arena_destroy(ar);
    g_simd_level = saved;

    ASSERT_TRUE(o0 == 58.0f);
    ASSERT_TRUE(o1 == 64.0f);
    ASSERT_TRUE(o2 == 139.0f);
    ASSERT_TRUE(o3 == 154.0f);
}

// identity: A @ I == A (sanity edge)
UTEST(avx_matmul, times_identity) {
    if(!__builtin_cpu_supports("avx"))
        return;
    SimdLevel saved = g_simd_level;
    pico_init();
    g_simd_level = SIMD_AVX;

    struct Arena* ar = arena_init(1 << 16);
    arena_ctx_push(ar);

    int64_t s[] = {2, 2};
    struct PicoTensor* a = pico_param(s, 2);
    struct PicoTensor* b = pico_param(s, 2);
    float av[] = {1, 2, 3, 4};
    for(int i = 0; i < 4; i++)
        a->data[i] = av[i];
    b->data[0] = 1;
    b->data[1] = 0;
    b->data[2] = 0;
    b->data[3] = 1;

    struct PicoTensor* out = pico_matmul(a, b);
    float o0 = out->data[0], o1 = out->data[1], o2 = out->data[2], o3 = out->data[3];

    pico_free(a);
    pico_free(b);
    arena_ctx_pop();
    arena_destroy(ar);
    g_simd_level = saved;

    ASSERT_TRUE(o0 == 1.0f);
    ASSERT_TRUE(o1 == 2.0f);
    ASSERT_TRUE(o2 == 3.0f);
    ASSERT_TRUE(o3 == 4.0f);
}

// EDGE: wide columns (10 > 8) so the vectorized j-loop AND the tail both run.
// A=1x2 [2,3], B=2x10 with b[0][j]=j, b[1][j]=0 -> out[j] = 2*j
UTEST(avx_matmul, wide_columns_10) {
    if(!__builtin_cpu_supports("avx"))
        return;
    SimdLevel saved = g_simd_level;
    pico_init();
    g_simd_level = SIMD_AVX;

    struct Arena* ar = arena_init(1 << 16);
    arena_ctx_push(ar);

    int64_t sa[] = {1, 2};
    int64_t sb[] = {2, 10};
    struct PicoTensor* a = pico_param(sa, 2);
    struct PicoTensor* b = pico_param(sb, 2);
    a->data[0] = 2;
    a->data[1] = 3;
    for(int j = 0; j < 10; j++) {
        b->data[j] = (float)j;   // row 0
        b->data[10 + j] = 0.0f;  // row 1
    }

    struct PicoTensor* out = pico_matmul(a, b);
    float o0 = out->data[0], o7 = out->data[7], o8 = out->data[8], o9 = out->data[9];

    pico_tensor_print(out);

    pico_free(a);
    pico_free(b);
    arena_ctx_pop();
    arena_destroy(ar);
    g_simd_level = saved;

    ASSERT_TRUE(o0 == 0.0f);   // 2*0
    ASSERT_TRUE(o7 == 14.0f);  // 2*7 (vector region)
    ASSERT_TRUE(o8 == 16.0f);  // 2*8 (tail region)
    ASSERT_TRUE(o9 == 18.0f);  // 2*9
}

// EDGE: 1x1 @ 1x1 = single element. [3] @ [4] = [12]
UTEST(avx_matmul, single_1x1) {
    if(!__builtin_cpu_supports("avx"))
        return;
    SimdLevel saved = g_simd_level;
    pico_init();
    g_simd_level = SIMD_AVX;

    struct Arena* ar = arena_init(1 << 16);
    arena_ctx_push(ar);

    int64_t s[] = {1, 1};
    struct PicoTensor* a = pico_param(s, 2);
    struct PicoTensor* b = pico_param(s, 2);
    a->data[0] = 3;
    b->data[0] = 4;

    struct PicoTensor* out = pico_matmul(a, b);
    float o0 = out->data[0];

    pico_free(a);
    pico_free(b);
    arena_ctx_pop();
    arena_destroy(ar);
    g_simd_level = saved;

    ASSERT_TRUE(o0 == 12.0f);
}

// EDGE: 1x3 @ 3x1 = 1x1 (a dot product; inner dim 3, output single). = 32
UTEST(avx_matmul, row_dot_col) {
    if(!__builtin_cpu_supports("avx"))
        return;
    SimdLevel saved = g_simd_level;
    pico_init();
    g_simd_level = SIMD_AVX;

    struct Arena* ar = arena_init(1 << 16);
    arena_ctx_push(ar);

    int64_t sa[] = {1, 3};
    int64_t sb[] = {3, 1};
    struct PicoTensor* a = pico_param(sa, 2);
    struct PicoTensor* b = pico_param(sb, 2);
    a->data[0] = 1;
    a->data[1] = 2;
    a->data[2] = 3;
    b->data[0] = 4;
    b->data[1] = 5;
    b->data[2] = 6;

    struct PicoTensor* out = pico_matmul(a, b);
    float o0 = out->data[0];

    pico_free(a);
    pico_free(b);
    arena_ctx_pop();
    arena_destroy(ar);
    g_simd_level = saved;

    ASSERT_TRUE(o0 == 32.0f);  // 1*4 + 2*5 + 3*6
}
