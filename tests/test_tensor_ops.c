#include <stdint.h>

#include "pico.h"
#include "utest.h"

UTEST(tensor_ops_view, reshapes_arena_tensor_without_copying_data) {
    struct PicoContext* ctx = pico_init_verbose(false);

    int64_t shape[] = {2, 3};
    float values[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
    struct PicoTensor* tensor = pico_tensor_from_data(ctx, shape, 2, values);

    float* old_data = tensor->data;
    int64_t* old_shape = tensor->shape;
    int64_t* old_strides = tensor->strides;

    int64_t view_shape[] = {3, 2};
    pico_view(ctx, tensor, view_shape, 2);

    ASSERT_TRUE(tensor->data == old_data);
    ASSERT_TRUE(tensor->shape != old_shape);
    ASSERT_TRUE(tensor->strides != old_strides);
    ASSERT_EQ(tensor->ndim, 2);
    ASSERT_EQ(tensor->numel, 6);
    ASSERT_EQ(tensor->shape[0], (int64_t)3);
    ASSERT_EQ(tensor->shape[1], (int64_t)2);
    ASSERT_EQ(tensor->strides[0], (int64_t)2);
    ASSERT_EQ(tensor->strides[1], (int64_t)1);
    ASSERT_NEAR(tensor->data[0], 1.0f, 1e-6f);
    ASSERT_NEAR(tensor->data[5], 6.0f, 1e-6f);

    pico_shutdown(ctx);
}

UTEST(tensor_ops_view, can_increase_rank_when_numel_matches) {
    struct PicoContext* ctx = pico_init_verbose(false);

    int64_t shape[] = {2, 6};
    float values[] = {
        1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f,
        7.0f, 8.0f, 9.0f, 10.0f, 11.0f, 12.0f,
    };
    struct PicoTensor* tensor = pico_tensor_from_data(ctx, shape, 2, values);

    int64_t view_shape[] = {2, 3, 2};
    pico_view(ctx, tensor, view_shape, 3);

    ASSERT_EQ(tensor->ndim, 3);
    ASSERT_EQ(tensor->numel, 12);
    ASSERT_EQ(tensor->shape[0], (int64_t)2);
    ASSERT_EQ(tensor->shape[1], (int64_t)3);
    ASSERT_EQ(tensor->shape[2], (int64_t)2);
    ASSERT_EQ(tensor->strides[0], (int64_t)6);
    ASSERT_EQ(tensor->strides[1], (int64_t)2);
    ASSERT_EQ(tensor->strides[2], (int64_t)1);

    pico_shutdown(ctx);
}

UTEST(tensor_ops_view, rejects_shape_with_different_numel) {
    struct PicoContext* ctx = pico_init_verbose(false);

    int64_t shape[] = {2, 3};
    float values[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
    struct PicoTensor* tensor = pico_tensor_from_data(ctx, shape, 2, values);

    int64_t* old_shape = tensor->shape;
    int64_t* old_strides = tensor->strides;
    uint8_t old_ndim = tensor->ndim;

    int64_t bad_shape[] = {2, 4};
    pico_view(ctx, tensor, bad_shape, 2);

    ASSERT_TRUE(tensor->shape == old_shape);
    ASSERT_TRUE(tensor->strides == old_strides);
    ASSERT_EQ(tensor->ndim, old_ndim);
    ASSERT_EQ(tensor->numel, 6);
    ASSERT_EQ(tensor->shape[0], (int64_t)2);
    ASSERT_EQ(tensor->shape[1], (int64_t)3);

    pico_shutdown(ctx);
}

UTEST(tensor_ops_view, rejects_heap_params_for_now) {
    struct PicoContext* ctx = pico_init_verbose(false);

    int64_t shape[] = {2, 3};
    struct PicoTensor* tensor = pico_param(ctx, shape, 2);

    int64_t* old_shape = tensor->shape;
    int64_t* old_strides = tensor->strides;
    uint8_t old_ndim = tensor->ndim;

    int64_t view_shape[] = {3, 2};
    pico_view(ctx, tensor, view_shape, 2);

    ASSERT_TRUE(tensor->shape == old_shape);
    ASSERT_TRUE(tensor->strides == old_strides);
    ASSERT_EQ(tensor->ndim, old_ndim);
    ASSERT_EQ(tensor->numel, 6);
    ASSERT_EQ(tensor->shape[0], (int64_t)2);
    ASSERT_EQ(tensor->shape[1], (int64_t)3);

    pico_shutdown(ctx);
}

UTEST(tensor_ops_permute, transposes_2d_into_contiguous_data) {
    struct PicoContext* ctx = pico_init_verbose(false);

    int64_t shape[] = {2, 3};
    float values[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
    struct PicoTensor* tensor = pico_tensor_from_data(ctx, shape, 2, values);
    float* old_data = tensor->data;

    int64_t axes[] = {1, 0};
    pico_permute(ctx, tensor, axes);

    ASSERT_TRUE(tensor->data != old_data);
    ASSERT_EQ(tensor->ndim, 2);
    ASSERT_EQ(tensor->numel, 6);
    ASSERT_EQ(tensor->shape[0], (int64_t)3);
    ASSERT_EQ(tensor->shape[1], (int64_t)2);
    ASSERT_EQ(tensor->strides[0], (int64_t)2);
    ASSERT_EQ(tensor->strides[1], (int64_t)1);

    float expected[] = {1.0f, 4.0f, 2.0f, 5.0f, 3.0f, 6.0f};
    for(int i = 0; i < tensor->numel; i++) {
        ASSERT_NEAR(tensor->data[i], expected[i], 1e-6f);
    }

    pico_shutdown(ctx);
}

UTEST(tensor_ops_permute, swaps_sequence_and_head_dims_for_attention_layout) {
    struct PicoContext* ctx = pico_init_verbose(false);

    int64_t shape[] = {1, 2, 2, 3};
    float values[] = {
        0.0f, 1.0f, 2.0f,
        3.0f, 4.0f, 5.0f,
        6.0f, 7.0f, 8.0f,
        9.0f, 10.0f, 11.0f,
    };
    struct PicoTensor* tensor = pico_tensor_from_data(ctx, shape, 4, values);

    int64_t axes[] = {0, 2, 1, 3};
    pico_permute(ctx, tensor, axes);

    ASSERT_EQ(tensor->ndim, 4);
    ASSERT_EQ(tensor->shape[0], (int64_t)1);
    ASSERT_EQ(tensor->shape[1], (int64_t)2);
    ASSERT_EQ(tensor->shape[2], (int64_t)2);
    ASSERT_EQ(tensor->shape[3], (int64_t)3);
    ASSERT_EQ(tensor->strides[0], (int64_t)12);
    ASSERT_EQ(tensor->strides[1], (int64_t)6);
    ASSERT_EQ(tensor->strides[2], (int64_t)3);
    ASSERT_EQ(tensor->strides[3], (int64_t)1);

    float expected[] = {
        0.0f, 1.0f, 2.0f,
        6.0f, 7.0f, 8.0f,
        3.0f, 4.0f, 5.0f,
        9.0f, 10.0f, 11.0f,
    };
    for(int i = 0; i < tensor->numel; i++) {
        ASSERT_NEAR(tensor->data[i], expected[i], 1e-6f);
    }

    pico_shutdown(ctx);
}

UTEST(tensor_ops_permute, reverses_3d_axes) {
    struct PicoContext* ctx = pico_init_verbose(false);

    int64_t shape[] = {2, 3, 4};
    float values[24];
    for(int i = 0; i < 24; i++) {
        values[i] = (float)i;
    }
    struct PicoTensor* tensor = pico_tensor_from_data(ctx, shape, 3, values);

    int64_t axes[] = {2, 1, 0};
    pico_permute(ctx, tensor, axes);

    ASSERT_EQ(tensor->shape[0], (int64_t)4);
    ASSERT_EQ(tensor->shape[1], (int64_t)3);
    ASSERT_EQ(tensor->shape[2], (int64_t)2);
    ASSERT_EQ(tensor->strides[0], (int64_t)6);
    ASSERT_EQ(tensor->strides[1], (int64_t)2);
    ASSERT_EQ(tensor->strides[2], (int64_t)1);

    // old [1,0,2] has value 14. after [2,1,0], it lives at new [2,0,1].
    ASSERT_NEAR(tensor->data[13], 14.0f, 1e-6f);
    // old [0,2,3] has value 11. after [2,1,0], it lives at new [3,2,0].
    ASSERT_NEAR(tensor->data[22], 11.0f, 1e-6f);

    pico_shutdown(ctx);
}

UTEST(tensor_ops_permute, rejects_invalid_axes_without_mutating_tensor) {
    struct PicoContext* ctx = pico_init_verbose(false);

    int64_t shape[] = {2, 3, 4};
    float values[24];
    for(int i = 0; i < 24; i++) {
        values[i] = (float)i;
    }
    struct PicoTensor* tensor = pico_tensor_from_data(ctx, shape, 3, values);

    int64_t* old_shape = tensor->shape;
    int64_t* old_strides = tensor->strides;
    float* old_data = tensor->data;

    int64_t axes[] = {0, 0, 2};
    pico_permute(ctx, tensor, axes);

    ASSERT_TRUE(tensor->shape == old_shape);
    ASSERT_TRUE(tensor->strides == old_strides);
    ASSERT_TRUE(tensor->data == old_data);
    ASSERT_EQ(tensor->shape[0], (int64_t)2);
    ASSERT_EQ(tensor->shape[1], (int64_t)3);
    ASSERT_EQ(tensor->shape[2], (int64_t)4);

    pico_shutdown(ctx);
}

UTEST(tensor_ops_softmax, returns_row_wise_softmax_for_dim_1) {
    struct PicoContext* ctx = pico_init_verbose(false);

    int64_t shape[] = {2, 3};
    float values[] = {1.0f, 2.0f, 3.0f, 1.0f, 1.0f, 1.0f};
    struct PicoTensor* tensor = pico_tensor_from_data(ctx, shape, 2, values);

    struct PicoTensor* out = pico_softmax(ctx, tensor, 1);

    ASSERT_TRUE(out != NULL);
    ASSERT_TRUE(out != tensor);
    ASSERT_EQ(out->ndim, 2);
    ASSERT_EQ(out->shape[0], (int64_t)2);
    ASSERT_EQ(out->shape[1], (int64_t)3);
    ASSERT_EQ(out->numel, 6);

    ASSERT_NEAR(out->data[0], 0.09003057f, 1e-5f);
    ASSERT_NEAR(out->data[1], 0.24472847f, 1e-5f);
    ASSERT_NEAR(out->data[2], 0.66524096f, 1e-5f);
    ASSERT_NEAR(out->data[3], 0.33333334f, 1e-5f);
    ASSERT_NEAR(out->data[4], 0.33333334f, 1e-5f);
    ASSERT_NEAR(out->data[5], 0.33333334f, 1e-5f);

    pico_shutdown(ctx);
}

UTEST(tensor_ops_softmax, returns_column_wise_softmax_for_dim_0) {
    struct PicoContext* ctx = pico_init_verbose(false);

    int64_t shape[] = {2, 3};
    float values[] = {1.0f, 2.0f, 3.0f, 2.0f, 2.0f, 1.0f};
    struct PicoTensor* tensor = pico_tensor_from_data(ctx, shape, 2, values);

    struct PicoTensor* out = pico_softmax(ctx, tensor, 0);

    ASSERT_TRUE(out != NULL);
    ASSERT_EQ(out->shape[0], (int64_t)2);
    ASSERT_EQ(out->shape[1], (int64_t)3);

    ASSERT_NEAR(out->data[0], 0.26894143f, 1e-5f);
    ASSERT_NEAR(out->data[3], 0.73105860f, 1e-5f);
    ASSERT_NEAR(out->data[1], 0.5f, 1e-5f);
    ASSERT_NEAR(out->data[4], 0.5f, 1e-5f);
    ASSERT_NEAR(out->data[2], 0.88079708f, 1e-5f);
    ASSERT_NEAR(out->data[5], 0.11920292f, 1e-5f);

    pico_shutdown(ctx);
}

UTEST(tensor_ops_softmax, rejects_dim_out_of_range) {
    struct PicoContext* ctx = pico_init_verbose(false);

    int64_t shape[] = {2, 3};
    float values[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
    struct PicoTensor* tensor = pico_tensor_from_data(ctx, shape, 2, values);

    struct PicoTensor* out = pico_softmax(ctx, tensor, 2);

    ASSERT_TRUE(out == NULL);
    pico_shutdown(ctx);
}

UTEST(tensor_ops_softmax, reads_permuted_tensor_layout) {
    struct PicoContext* ctx = pico_init_verbose(false);

    int64_t shape[] = {2, 3};
    float values[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
    struct PicoTensor* tensor = pico_tensor_from_data(ctx, shape, 2, values);

    pico_transpose_2d(tensor);
    struct PicoTensor* out = pico_softmax(ctx, tensor, 1);

    ASSERT_TRUE(out != NULL);
    ASSERT_EQ(out->shape[0], (int64_t)3);
    ASSERT_EQ(out->shape[1], (int64_t)2);

    ASSERT_NEAR(out->data[0], 0.04742587f, 1e-5f);
    ASSERT_NEAR(out->data[1], 0.95257413f, 1e-5f);
    ASSERT_NEAR(out->data[2], 0.04742587f, 1e-5f);
    ASSERT_NEAR(out->data[3], 0.95257413f, 1e-5f);
    ASSERT_NEAR(out->data[4], 0.04742587f, 1e-5f);
    ASSERT_NEAR(out->data[5], 0.95257413f, 1e-5f);

    pico_shutdown(ctx);
}
