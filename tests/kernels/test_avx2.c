/*
 * Tests for the AVX2 element-wise kernels.
 * NOTE: no UTEST_MAIN here, test_basic.c owns main + UTEST_STATE.
 *
 * The rest of the suite runs with g_simd_level == SIMD_NONE (nothing calls
 * pico_init), so it exercises the SCALAR path. These tests deliberately FORCE
 * g_simd_level = SIMD_AVX2 to route pico_add through pico_add_cpu_avx2_fp32 —
 * and RESTORE the old level before asserting, so they never disturb the other
 * tests (the global doesn't leak, even on a failed assert).
 *
 * Guarded by __builtin_cpu_supports so a non-AVX2 machine skips instead of SIGILL.
 */
#include "arena.h"
#include "global.h"
#include "ops.h"
#include "tensor.h"
#include "utest.h"

// --- shape-equality helper (pins the sizeof bug; independent of AVX2) --------

// identical shapes must compare equal
UTEST(shapes_equal, identical_equal) {
    int64_t s[] = {2, 3};
    struct PicoTensor* a = pico_param(s, 2);
    struct PicoTensor* b = pico_param(s, 2);

    bool eq = pico_tensor_shapes_are_equal(a, b);

    pico_free(a);
    pico_free(b);
    ASSERT_TRUE(eq);
}

// FAILS until the memcmp uses sizeof(int64_t) not sizeof(float): with sizeof(float)
// only ~half the bytes are compared, so for 2-D shapes only shape[0] is checked and
// (2,3) vs (2,5) wrongly reports equal.
UTEST(shapes_equal, different_2d_not_equal) {
    int64_t sa[] = {2, 3};
    int64_t sb[] = {2, 5};
    struct PicoTensor* a = pico_param(sa, 2);
    struct PicoTensor* b = pico_param(sb, 2);

    bool eq = pico_tensor_shapes_are_equal(a, b);

    pico_free(a);
    pico_free(b);
    ASSERT_FALSE(eq);  // currently true -> fails here
}

// --- AVX2 add: force the SIMD level, run pico_add, restore --------------------

// 16 elements = exactly TWO full vector iterations, no tail.
// a[i]=i, b[i]=10i -> out[i]=11i
UTEST(kernel_avx2, add_size16_two_vectors) {
    if(!__builtin_cpu_supports("avx2")) return;  // skip on non-AVX2 machines

    SimdLevel saved = g_simd_level;
    pico_init();
    g_simd_level = SIMD_AVX2;  // force the AVX2 path

    struct Arena* ar = arena_init(1 << 16);
    arena_ctx_push(ar);

    int64_t s[] = {16};
    struct PicoTensor* a = pico_param(s, 1);
    struct PicoTensor* b = pico_param(s, 1);
    for(int i = 0; i < 16; i++) {
        a->data[i] = (float)i;
        b->data[i] = (float)(i * 10);
    }

    struct PicoTensor* out = pico_add(a, b);

    // capture across BOTH vector iterations before teardown
    float o0 = out->data[0];    // 0
    float o7 = out->data[7];    // 77   (end of vector 1)
    float o8 = out->data[8];    // 88   (start of vector 2)
    float o15 = out->data[15];  // 165  (end of vector 2)

    pico_free(a);
    pico_free(b);
    arena_ctx_pop();
    arena_destroy(ar);
    g_simd_level = saved;  // restore BEFORE asserting

    ASSERT_TRUE(o0 == 0.0f);
    ASSERT_TRUE(o7 == 77.0f);
    ASSERT_TRUE(o8 == 88.0f);
    ASSERT_TRUE(o15 == 165.0f);
}

// 19 elements = two full vectors [0..15] + a 3-element SCALAR TAIL [16,17,18].
// a[i]=i, b[i]=i -> out[i]=2i
UTEST(kernel_avx2, add_size19_vector_plus_tail) {
    if(!__builtin_cpu_supports("avx2")) return;

    SimdLevel saved = g_simd_level;
    pico_init();
    g_simd_level = SIMD_AVX2;

    struct Arena* ar = arena_init(1 << 16);
    arena_ctx_push(ar);

    int64_t s[] = {19};
    struct PicoTensor* a = pico_param(s, 1);
    struct PicoTensor* b = pico_param(s, 1);
    for(int i = 0; i < 19; i++) {
        a->data[i] = (float)i;
        b->data[i] = (float)i;
    }

    struct PicoTensor* out = pico_add(a, b);

    float o0 = out->data[0];    // 0   (vector)
    float o15 = out->data[15];  // 30  (last vector elem)
    float o16 = out->data[16];  // 32  (tail)
    float o17 = out->data[17];  // 34  (tail)
    float o18 = out->data[18];  // 36  (tail)

    pico_free(a);
    pico_free(b);
    arena_ctx_pop();
    arena_destroy(ar);
    g_simd_level = saved;

    ASSERT_TRUE(o0 == 0.0f);
    ASSERT_TRUE(o15 == 30.0f);
    ASSERT_TRUE(o16 == 32.0f);  // tail correctness
    ASSERT_TRUE(o17 == 34.0f);
    ASSERT_TRUE(o18 == 36.0f);
}

// 5 elements < 8: the vector loop is skipped entirely (bound i <= size-8 is
// 0 <= -3, false), so ONLY the scalar tail runs. proves small tensors are safe.
// a[i]=i, b[i]=2i -> out[i]=3i
UTEST(kernel_avx2, add_size5_only_tail) {
    if(!__builtin_cpu_supports("avx2")) return;

    SimdLevel saved = g_simd_level;
    pico_init();
    g_simd_level = SIMD_AVX2;

    struct Arena* ar = arena_init(1 << 16);
    arena_ctx_push(ar);

    int64_t s[] = {5};
    struct PicoTensor* a = pico_param(s, 1);
    struct PicoTensor* b = pico_param(s, 1);
    for(int i = 0; i < 5; i++) {
        a->data[i] = (float)i;
        b->data[i] = (float)(i * 2);
    }

    struct PicoTensor* out = pico_add(a, b);

    float o0 = out->data[0];  // 0
    float o4 = out->data[4];  // 12

    pico_free(a);
    pico_free(b);
    arena_ctx_pop();
    arena_destroy(ar);
    g_simd_level = saved;

    ASSERT_TRUE(o0 == 0.0f);
    ASSERT_TRUE(o4 == 12.0f);
}

// --- AVX2 sub / mul: same force-and-restore pattern -------------------------
// pico_sub_cpu / pico_mul_cpu dispatch g_simd_level == SIMD_AVX2 to their
// *_cpu_avx2_fp32 kernels, so forcing the level here drives the real AVX2 path.

// sub, 16 elems (two full vectors). a[i]=3i, b[i]=i -> out[i]=2i
UTEST(kernel_avx2, sub_size16_two_vectors) {
    if(!__builtin_cpu_supports("avx2")) return;

    SimdLevel saved = g_simd_level;
    pico_init();
    g_simd_level = SIMD_AVX2;

    struct Arena* ar = arena_init(1 << 16);
    arena_ctx_push(ar);

    int64_t s[] = {16};
    struct PicoTensor* a = pico_param(s, 1);
    struct PicoTensor* b = pico_param(s, 1);
    for(int i = 0; i < 16; i++) {
        a->data[i] = (float)(i * 3);
        b->data[i] = (float)i;
    }

    struct PicoTensor* out = pico_sub(a, b);
    float o0 = out->data[0];    // 0
    float o7 = out->data[7];    // 14
    float o8 = out->data[8];    // 16
    float o15 = out->data[15];  // 30

    pico_free(a);
    pico_free(b);
    arena_ctx_pop();
    arena_destroy(ar);
    g_simd_level = saved;

    ASSERT_TRUE(o0 == 0.0f);
    ASSERT_TRUE(o7 == 14.0f);
    ASSERT_TRUE(o8 == 16.0f);
    ASSERT_TRUE(o15 == 30.0f);
}

// sub, 19 elems (vector + 3-elem scalar tail). a[i]=3i, b[i]=i -> out[i]=2i
UTEST(kernel_avx2, sub_size19_vector_plus_tail) {
    if(!__builtin_cpu_supports("avx2")) return;

    SimdLevel saved = g_simd_level;
    pico_init();
    g_simd_level = SIMD_AVX2;

    struct Arena* ar = arena_init(1 << 16);
    arena_ctx_push(ar);

    int64_t s[] = {19};
    struct PicoTensor* a = pico_param(s, 1);
    struct PicoTensor* b = pico_param(s, 1);
    for(int i = 0; i < 19; i++) {
        a->data[i] = (float)(i * 3);
        b->data[i] = (float)i;
    }

    struct PicoTensor* out = pico_sub(a, b);
    float o15 = out->data[15];  // 30 (vector)
    float o16 = out->data[16];  // 32 (tail)
    float o18 = out->data[18];  // 36 (tail)

    pico_free(a);
    pico_free(b);
    arena_ctx_pop();
    arena_destroy(ar);
    g_simd_level = saved;

    ASSERT_TRUE(o15 == 30.0f);
    ASSERT_TRUE(o16 == 32.0f);
    ASSERT_TRUE(o18 == 36.0f);
}

// mul, 16 elems (two full vectors). a[i]=i, b[i]=i -> out[i]=i^2
UTEST(kernel_avx2, mul_size16_two_vectors) {
    if(!__builtin_cpu_supports("avx2")) return;

    SimdLevel saved = g_simd_level;
    pico_init();
    g_simd_level = SIMD_AVX2;

    struct Arena* ar = arena_init(1 << 16);
    arena_ctx_push(ar);

    int64_t s[] = {16};
    struct PicoTensor* a = pico_param(s, 1);
    struct PicoTensor* b = pico_param(s, 1);
    for(int i = 0; i < 16; i++) {
        a->data[i] = (float)i;
        b->data[i] = (float)i;
    }

    struct PicoTensor* out = pico_mul(a, b);
    float o0 = out->data[0];    // 0
    float o7 = out->data[7];    // 49
    float o8 = out->data[8];    // 64
    float o15 = out->data[15];  // 225

    pico_free(a);
    pico_free(b);
    arena_ctx_pop();
    arena_destroy(ar);
    g_simd_level = saved;

    ASSERT_TRUE(o0 == 0.0f);
    ASSERT_TRUE(o7 == 49.0f);
    ASSERT_TRUE(o8 == 64.0f);
    ASSERT_TRUE(o15 == 225.0f);
}

// mul, 19 elems (vector + tail). a[i]=i, b[i]=i -> out[i]=i^2
UTEST(kernel_avx2, mul_size19_vector_plus_tail) {
    if(!__builtin_cpu_supports("avx2")) return;

    SimdLevel saved = g_simd_level;
    pico_init();
    g_simd_level = SIMD_AVX2;

    struct Arena* ar = arena_init(1 << 16);
    arena_ctx_push(ar);

    int64_t s[] = {19};
    struct PicoTensor* a = pico_param(s, 1);
    struct PicoTensor* b = pico_param(s, 1);
    for(int i = 0; i < 19; i++) {
        a->data[i] = (float)i;
        b->data[i] = (float)i;
    }

    struct PicoTensor* out = pico_mul(a, b);
    float o15 = out->data[15];  // 225 (vector)
    float o16 = out->data[16];  // 256 (tail)
    float o18 = out->data[18];  // 324 (tail)

    pico_free(a);
    pico_free(b);
    arena_ctx_pop();
    arena_destroy(ar);
    g_simd_level = saved;

    ASSERT_TRUE(o15 == 225.0f);
    ASSERT_TRUE(o16 == 256.0f);
    ASSERT_TRUE(o18 == 324.0f);
}
