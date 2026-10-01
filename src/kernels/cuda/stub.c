// INFO: cpu-only fallback for the cuda boundary.
// normal gcc builds still compile dispatch code that references pico_cuda_*
// symbols, even when no .cu kernels are linked. these weak definitions keep
// that build linkable and fail loudly if a cuda path is accidentally called.
// when libpico_cuda.a is linked, the real .cu definitions override these.

#include "cuda_ops.h"

#include <stdio.h>
#include <string.h>

#include "../../ctx.h"

#if defined(__GNUC__)
#define PICO_WEAK __attribute__((weak))
#else
#define PICO_WEAK
#endif

static bool pico_cuda_stub_missing(const char *op_name) {
    fprintf(stderr, "PicoCudaError: %s CUDA kernel is not linked yet\n", op_name);
    return false;
}

PICO_WEAK bool pico_cuda_tensor_alloc(struct PicoContext *ctx, struct PicoTensor *tensor) {
    (void)ctx;
    if(tensor == NULL) {
        return false;
    }

    // INFO: cpu-only build compatibility. we only mark the backend here; the
    // real .cu implementation will cudaMalloc data and grad.
    tensor->backend = PICO_BACKEND_CUDA;
    return true;
}

PICO_WEAK bool pico_cuda_tensor_to_cuda(struct PicoContext *ctx, struct PicoTensor *tensor) {
    (void)ctx;
    if(tensor == NULL) {
        return false;
    }

    // INFO: same fake-transfer behavior as before. real cuda will allocate and copy.
    tensor->backend = PICO_BACKEND_CUDA;
    return true;
}

PICO_WEAK bool pico_cuda_tensor_to_cpu(struct PicoContext *ctx, struct PicoTensor *tensor, float *old_data, float *old_grad) {
    (void)ctx;
    if(tensor == NULL) {
        return false;
    }

    if(old_data != NULL && tensor->data != NULL && old_data != tensor->data) {
        memcpy(tensor->data, old_data, tensor->numel * sizeof(float));
    }
    if(old_grad != NULL && tensor->grad != NULL && old_grad != tensor->grad) {
        memcpy(tensor->grad, old_grad, tensor->numel * sizeof(float));
    }
    tensor->backend = PICO_BACKEND_CPU;
    return true;
}

PICO_WEAK bool pico_cuda_add(struct PicoTensor *a, struct PicoTensor *b, struct PicoTensor *out) {
    (void)a;
    (void)b;
    (void)out;
    return pico_cuda_stub_missing("add");
}

PICO_WEAK bool pico_cuda_sub(struct PicoTensor *a, struct PicoTensor *b, struct PicoTensor *out) {
    (void)a;
    (void)b;
    (void)out;
    return pico_cuda_stub_missing("sub");
}

PICO_WEAK bool pico_cuda_mul(struct PicoTensor *a, struct PicoTensor *b, struct PicoTensor *out) {
    (void)a;
    (void)b;
    (void)out;
    return pico_cuda_stub_missing("mul");
}

PICO_WEAK bool pico_cuda_div(struct PicoTensor *a, struct PicoTensor *b, struct PicoTensor *out) {
    (void)a;
    (void)b;
    (void)out;
    return pico_cuda_stub_missing("div");
}

#define PICO_DEFINE_CUDA_UNARY_STUB(name)                                                                                                  \
    PICO_WEAK bool pico_cuda_##name(struct PicoTensor *a, struct PicoTensor *out) {                                                        \
        (void)a;                                                                                                                           \
        (void)out;                                                                                                                         \
        return pico_cuda_stub_missing(#name);                                                                                              \
    }

PICO_DEFINE_CUDA_UNARY_STUB(sqrt)
PICO_DEFINE_CUDA_UNARY_STUB(rsqrt)
PICO_DEFINE_CUDA_UNARY_STUB(sin)
PICO_DEFINE_CUDA_UNARY_STUB(cos)
PICO_DEFINE_CUDA_UNARY_STUB(tan)
PICO_DEFINE_CUDA_UNARY_STUB(tanh)
PICO_DEFINE_CUDA_UNARY_STUB(log)
PICO_DEFINE_CUDA_UNARY_STUB(relu)
PICO_DEFINE_CUDA_UNARY_STUB(sigmoid)

PICO_WEAK bool pico_cuda_matmul(struct PicoTensor *a, struct PicoTensor *b, struct PicoTensor *out) {
    (void)a;
    (void)b;
    (void)out;
    return pico_cuda_stub_missing("matmul");
}

PICO_WEAK bool pico_cuda_grouped_matmul(struct PicoTensor *a, struct PicoTensor *b, struct PicoTensor *out, int group_size) {
    (void)a;
    (void)b;
    (void)out;
    (void)group_size;
    return pico_cuda_stub_missing("grouped_matmul");
}

PICO_WEAK bool pico_cuda_softmax(struct PicoTensor *input, struct PicoTensor *out, uint8_t dim) {
    (void)input;
    (void)out;
    (void)dim;
    return pico_cuda_stub_missing("softmax");
}

PICO_WEAK bool pico_cuda_mean(struct PicoTensor *input, struct PicoTensor *out, int dim) {
    (void)input;
    (void)out;
    (void)dim;
    return pico_cuda_stub_missing("mean");
}

PICO_WEAK bool pico_cuda_rmsnorm(struct PicoTensor *input, struct PicoTensor *weight, struct PicoTensor *out, float eps) {
    (void)input;
    (void)weight;
    (void)out;
    (void)eps;
    return pico_cuda_stub_missing("rmsnorm");
}

PICO_WEAK bool pico_cuda_swiglu(struct PicoTensor *x, struct PicoTensor *gate, struct PicoTensor *out) {
    (void)x;
    (void)gate;
    (void)out;
    return pico_cuda_stub_missing("swiglu");
}

PICO_WEAK bool pico_cuda_fused_swiglu(struct PicoTensor *gate, struct PicoTensor *up, struct PicoTensor *mask,
                                      struct PicoTensor *out, float dropout_p, bool training, uint32_t seed) {
    (void)gate;
    (void)up;
    (void)mask;
    (void)out;
    (void)dropout_p;
    (void)training;
    (void)seed;
    return pico_cuda_stub_missing("fused_swiglu");
}

PICO_WEAK bool pico_cuda_embedding(struct PicoTensor *table, struct PicoTensor *input_indices, struct PicoTensor *out) {
    (void)table;
    (void)input_indices;
    (void)out;
    return pico_cuda_stub_missing("embedding");
}

PICO_WEAK bool pico_cuda_cross_entropy(struct PicoTensor *logits, struct PicoTensor *targets, struct PicoTensor *out, int reduction) {
    (void)logits;
    (void)targets;
    (void)out;
    (void)reduction;
    return pico_cuda_stub_missing("cross_entropy");
}
