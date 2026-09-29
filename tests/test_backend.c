#include <string.h>

#include "pico.h"
#include "utest.h"

UTEST(backend, name_cpu) {
    ASSERT_STREQ("CPU", pico_backend_name(PICO_BACKEND_CPU));
}

UTEST(backend, name_cuda) {
    ASSERT_STREQ("CUDA", pico_backend_name(PICO_BACKEND_CUDA));
}

UTEST(backend, name_unknown) {
    ASSERT_STREQ("UNKNOWN", pico_backend_name((PicoBackend)99));
}

UTEST(backend, require_same_backend_accepts_cpu_tensors) {
    struct PicoContext* ctx = pico_init_verbose(false);
    int64_t shape[] = {2};
    struct PicoTensor* a = pico_create_tensor(ctx, shape, 1);
    struct PicoTensor* b = pico_create_tensor(ctx, shape, 1);

    ASSERT_TRUE(pico_require_same_backend(a, b, "test"));

    pico_shutdown(ctx);
}

UTEST(backend, require_same_backend_rejects_mismatch) {
    struct PicoContext* ctx = pico_init_verbose(false);
    int64_t shape[] = {2};
    struct PicoTensor* a = pico_create_tensor(ctx, shape, 1);
    struct PicoTensor* b = pico_create_tensor_on(ctx, PICO_BACKEND_CUDA, shape, 1);

    ASSERT_FALSE(pico_require_same_backend(a, b, "test"));

    pico_shutdown(ctx);
}

UTEST(backend, require_cpu_backend_accepts_cpu) {
    ASSERT_TRUE(pico_require_cpu_backend(PICO_BACKEND_CPU, "test"));
}

UTEST(backend, require_cpu_backend_rejects_cuda) {
    ASSERT_FALSE(pico_require_cpu_backend(PICO_BACKEND_CUDA, "test"));
}

UTEST(backend_alloc, create_tensor_on_cpu_allocates_payload) {
    struct PicoContext* ctx = pico_init_verbose(false);
    int64_t shape[] = {2, 2};
    struct PicoTensor* t = pico_create_tensor_on(ctx, PICO_BACKEND_CPU, shape, 2);

    ASSERT_TRUE(t != NULL);
    ASSERT_EQ(t->kind, PICO_TENSOR_TEMP);
    ASSERT_EQ(t->backend, PICO_BACKEND_CPU);
    ASSERT_TRUE(t->data != NULL);
    ASSERT_TRUE(t->grad != NULL);

    pico_shutdown(ctx);
}

UTEST(backend_alloc, create_tensor_on_cuda_marks_without_payload) {
    struct PicoContext* ctx = pico_init_verbose(false);
    int64_t shape[] = {2, 2};
    struct PicoTensor* t = pico_create_tensor_on(ctx, PICO_BACKEND_CUDA, shape, 2);

    ASSERT_TRUE(t != NULL);
    ASSERT_EQ(t->kind, PICO_TENSOR_TEMP);
    ASSERT_EQ(t->backend, PICO_BACKEND_CUDA);
    ASSERT_TRUE(t->data == NULL);
    ASSERT_TRUE(t->grad == NULL);

    pico_shutdown(ctx);
}

UTEST(backend_alloc, param_on_cpu_allocates_payload_and_registers) {
    struct PicoContext* ctx = pico_init_verbose(false);
    int64_t shape[] = {3};
    struct PicoTensor* t = pico_param_on(ctx, PICO_BACKEND_CPU, shape, 1);

    ASSERT_TRUE(t != NULL);
    ASSERT_EQ(t->kind, PICO_TENSOR_PARAM);
    ASSERT_EQ(t->backend, PICO_BACKEND_CPU);
    ASSERT_TRUE(t->data != NULL);
    ASSERT_TRUE(t->grad != NULL);
    ASSERT_EQ(ctx->params.size, (size_t)1);

    pico_shutdown(ctx);
}

UTEST(backend_alloc, param_on_cuda_marks_without_payload_and_registers) {
    struct PicoContext* ctx = pico_init_verbose(false);
    int64_t shape[] = {3};
    struct PicoTensor* t = pico_param_on(ctx, PICO_BACKEND_CUDA, shape, 1);

    ASSERT_TRUE(t != NULL);
    ASSERT_EQ(t->kind, PICO_TENSOR_PARAM);
    ASSERT_EQ(t->backend, PICO_BACKEND_CUDA);
    ASSERT_TRUE(t->data == NULL);
    ASSERT_TRUE(t->grad == NULL);
    ASSERT_EQ(ctx->params.size, (size_t)1);

    pico_shutdown(ctx);
}

UTEST(backend_transfer, cpu_to_cpu_is_noop) {
    struct PicoContext* ctx = pico_init_verbose(false);
    int64_t shape[] = {2};
    struct PicoTensor* t = pico_tensor_from_data(ctx, shape, 1, (float[]){1.0f, 2.0f});
    float* old_data = t->data;

    ASSERT_TRUE(pico_tensor_to_backend(ctx, t, PICO_BACKEND_CPU));
    ASSERT_EQ(t->backend, PICO_BACKEND_CPU);
    ASSERT_TRUE(t->data == old_data);
    ASSERT_TRUE(t->data[1] == 2.0f);

    pico_shutdown(ctx);
}

UTEST(backend_transfer, cuda_to_cuda_is_noop) {
    struct PicoContext* ctx = pico_init_verbose(false);
    int64_t shape[] = {2};
    struct PicoTensor* t = pico_tensor_from_data_on(ctx, PICO_BACKEND_CUDA, shape, 1, (float[]){1.0f, 2.0f});
    float* old_data = t->data;

    ASSERT_TRUE(pico_tensor_to_backend(ctx, t, PICO_BACKEND_CUDA));
    ASSERT_EQ(t->backend, PICO_BACKEND_CUDA);
    ASSERT_TRUE(t->data == old_data);

    pico_shutdown(ctx);
}

UTEST(backend_transfer, cpu_to_cuda_marks_backend_and_preserves_data_pointer) {
    struct PicoContext* ctx = pico_init_verbose(false);
    int64_t shape[] = {2};
    struct PicoTensor* t = pico_tensor_from_data(ctx, shape, 1, (float[]){5.0f, 6.0f});
    float* old_data = t->data;

    ASSERT_TRUE(pico_tensor_to_backend(ctx, t, PICO_BACKEND_CUDA));
    ASSERT_EQ(t->backend, PICO_BACKEND_CUDA);
    ASSERT_TRUE(t->data == old_data);
    ASSERT_TRUE(t->data[0] == 5.0f);

    pico_shutdown(ctx);
}

UTEST(backend_transfer, cpu_to_cuda_preserves_grad_pointer_and_values) {
    struct PicoContext* ctx = pico_init_verbose(false);
    int64_t shape[] = {2};
    struct PicoTensor* t = pico_create_tensor(ctx, shape, 1);
    t->grad[0] = 3.0f;
    float* old_grad = t->grad;

    ASSERT_TRUE(pico_tensor_to_backend(ctx, t, PICO_BACKEND_CUDA));
    ASSERT_EQ(t->backend, PICO_BACKEND_CUDA);
    ASSERT_TRUE(t->grad == old_grad);
    ASSERT_TRUE(t->grad[0] == 3.0f);

    pico_shutdown(ctx);
}

UTEST(backend_transfer, cuda_to_cpu_allocates_new_data_pointer) {
    struct PicoContext* ctx = pico_init_verbose(false);
    int64_t shape[] = {2};
    struct PicoTensor* t = pico_tensor_from_data_on(ctx, PICO_BACKEND_CUDA, shape, 1, (float[]){7.0f, 8.0f});
    float* old_data = t->data;

    ASSERT_TRUE(pico_tensor_to_backend(ctx, t, PICO_BACKEND_CPU));
    ASSERT_EQ(t->backend, PICO_BACKEND_CPU);
    ASSERT_TRUE(t->data != old_data);
    ASSERT_TRUE(t->data[1] == 8.0f);

    pico_shutdown(ctx);
}

UTEST(backend_transfer, cuda_to_cpu_preserves_grad_values) {
    struct PicoContext* ctx = pico_init_verbose(false);
    int64_t shape[] = {2};
    struct PicoTensor* t = pico_create_tensor(ctx, shape, 1);
    t->grad[0] = 9.0f;
    ASSERT_TRUE(pico_tensor_to_backend(ctx, t, PICO_BACKEND_CUDA));

    ASSERT_TRUE(pico_tensor_to_backend(ctx, t, PICO_BACKEND_CPU));
    ASSERT_EQ(t->backend, PICO_BACKEND_CPU);
    ASSERT_TRUE(t->grad != NULL);
    ASSERT_TRUE(t->grad[0] == 9.0f);

    pico_shutdown(ctx);
}

UTEST(backend_transfer, from_data_on_cuda_preserves_shape_and_strides) {
    struct PicoContext* ctx = pico_init_verbose(false);
    int64_t shape[] = {2, 3};
    struct PicoTensor* t = pico_tensor_from_data_on(ctx, PICO_BACKEND_CUDA, shape, 2,
                                                    (float[]){1, 2, 3, 4, 5, 6});

    ASSERT_TRUE(t != NULL);
    ASSERT_EQ(t->backend, PICO_BACKEND_CUDA);
    ASSERT_EQ(t->shape[0], (int64_t)2);
    ASSERT_EQ(t->shape[1], (int64_t)3);
    ASSERT_EQ(t->strides[0], (int64_t)3);
    ASSERT_EQ(t->strides[1], (int64_t)1);

    pico_shutdown(ctx);
}

UTEST(backend_transfer, scalar_on_cuda_preserves_scalar_value) {
    struct PicoContext* ctx = pico_init_verbose(false);
    struct PicoTensor* t = pico_tensor_from_scalar_on(ctx, PICO_BACKEND_CUDA, 11.0f);

    ASSERT_TRUE(t != NULL);
    ASSERT_EQ(t->backend, PICO_BACKEND_CUDA);
    ASSERT_EQ(t->numel, 1);
    ASSERT_TRUE(t->data[0] == 11.0f);

    pico_shutdown(ctx);
}

UTEST(backend_ops, add_rejects_cpu_cuda_mismatch) {
    struct PicoContext* ctx = pico_init_verbose(false);
    int64_t shape[] = {2};
    struct PicoTensor* a = pico_tensor_from_data(ctx, shape, 1, (float[]){1, 2});
    struct PicoTensor* b = pico_tensor_from_data_on(ctx, PICO_BACKEND_CUDA, shape, 1, (float[]){3, 4});

    ASSERT_TRUE(pico_add(ctx, a, b) == NULL);

    pico_shutdown(ctx);
}

UTEST(backend_ops, add_rejects_cuda_cuda) {
    struct PicoContext* ctx = pico_init_verbose(false);
    int64_t shape[] = {2};
    struct PicoTensor* a = pico_tensor_from_data_on(ctx, PICO_BACKEND_CUDA, shape, 1, (float[]){1, 2});
    struct PicoTensor* b = pico_tensor_from_data_on(ctx, PICO_BACKEND_CUDA, shape, 1, (float[]){3, 4});

    ASSERT_TRUE(pico_add(ctx, a, b) == NULL);

    pico_shutdown(ctx);
}

UTEST(backend_ops, sub_rejects_cuda_cuda) {
    struct PicoContext* ctx = pico_init_verbose(false);
    int64_t shape[] = {2};
    struct PicoTensor* a = pico_tensor_from_data_on(ctx, PICO_BACKEND_CUDA, shape, 1, (float[]){1, 2});
    struct PicoTensor* b = pico_tensor_from_data_on(ctx, PICO_BACKEND_CUDA, shape, 1, (float[]){3, 4});

    ASSERT_TRUE(pico_sub(ctx, a, b) == NULL);

    pico_shutdown(ctx);
}

UTEST(backend_ops, mul_rejects_cuda_cuda) {
    struct PicoContext* ctx = pico_init_verbose(false);
    int64_t shape[] = {2};
    struct PicoTensor* a = pico_tensor_from_data_on(ctx, PICO_BACKEND_CUDA, shape, 1, (float[]){1, 2});
    struct PicoTensor* b = pico_tensor_from_data_on(ctx, PICO_BACKEND_CUDA, shape, 1, (float[]){3, 4});

    ASSERT_TRUE(pico_mul(ctx, a, b) == NULL);

    pico_shutdown(ctx);
}

UTEST(backend_ops, div_rejects_cuda_cuda) {
    struct PicoContext* ctx = pico_init_verbose(false);
    int64_t shape[] = {2};
    struct PicoTensor* a = pico_tensor_from_data_on(ctx, PICO_BACKEND_CUDA, shape, 1, (float[]){1, 2});
    struct PicoTensor* b = pico_tensor_from_data_on(ctx, PICO_BACKEND_CUDA, shape, 1, (float[]){3, 4});

    ASSERT_TRUE(pico_div(ctx, a, b) == NULL);

    pico_shutdown(ctx);
}

UTEST(backend_ops, matmul_rejects_cuda_cuda) {
    struct PicoContext* ctx = pico_init_verbose(false);
    int64_t shape[] = {2, 2};
    struct PicoTensor* a = pico_tensor_from_data_on(ctx, PICO_BACKEND_CUDA, shape, 2, (float[]){1, 2, 3, 4});
    struct PicoTensor* b = pico_tensor_from_data_on(ctx, PICO_BACKEND_CUDA, shape, 2, (float[]){1, 0, 0, 1});

    ASSERT_TRUE(pico_matmul(ctx, a, b) == NULL);

    pico_shutdown(ctx);
}

UTEST(backend_ops, relu_rejects_cuda) {
    struct PicoContext* ctx = pico_init_verbose(false);
    int64_t shape[] = {2};
    struct PicoTensor* t = pico_tensor_from_data_on(ctx, PICO_BACKEND_CUDA, shape, 1, (float[]){-1, 2});

    ASSERT_TRUE(pico_relu(ctx, t) == NULL);

    pico_shutdown(ctx);
}

UTEST(backend_ops, swiglu_rejects_cuda) {
    struct PicoContext* ctx = pico_init_verbose(false);
    int64_t shape[] = {2};
    struct PicoTensor* x = pico_tensor_from_data_on(ctx, PICO_BACKEND_CUDA, shape, 1, (float[]){-1, 2});
    struct PicoTensor* gate = pico_tensor_from_data_on(ctx, PICO_BACKEND_CUDA, shape, 1, (float[]){3, 4});

    ASSERT_TRUE(pico_swiglu(ctx, x, gate) == NULL);

    pico_shutdown(ctx);
}

UTEST(backend_ops, softmax_rejects_cuda) {
    struct PicoContext* ctx = pico_init_verbose(false);
    int64_t shape[] = {2};
    struct PicoTensor* t = pico_tensor_from_data_on(ctx, PICO_BACKEND_CUDA, shape, 1, (float[]){1, 2});

    ASSERT_TRUE(pico_softmax(ctx, t, 0) == NULL);

    pico_shutdown(ctx);
}

UTEST(backend_ops, view_rejects_cuda_without_mutating_metadata) {
    struct PicoContext* ctx = pico_init_verbose(false);
    int64_t shape[] = {2, 2};
    struct PicoTensor* t = pico_tensor_from_data_on(ctx, PICO_BACKEND_CUDA, shape, 2, (float[]){1, 2, 3, 4});
    int64_t* old_shape = t->shape;
    int64_t* old_strides = t->strides;
    int64_t view_shape[] = {4};

    pico_view(ctx, t, view_shape, 1);
    ASSERT_TRUE(t->shape == old_shape);
    ASSERT_TRUE(t->strides == old_strides);
    ASSERT_EQ(t->ndim, 2);

    pico_shutdown(ctx);
}

UTEST(backend_ops, clone_rejects_cuda) {
    struct PicoContext* ctx = pico_init_verbose(false);
    int64_t shape[] = {2};
    struct PicoTensor* t = pico_tensor_from_data_on(ctx, PICO_BACKEND_CUDA, shape, 1, (float[]){1, 2});

    ASSERT_TRUE(pico_clone(ctx, t) == NULL);

    pico_shutdown(ctx);
}
