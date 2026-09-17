
#include "cuda_backend.h"
#include "gemm/gemm.cuh"

extern "C" void pico_cuda_matmul(PicoTensor *A, PicoTensor *B, PicoTensor *C) {
    int M = A->shape[0];
    int K = A->shape[1];
    int N = B->shape[1];

    float *a = A->data;
    float *b = B->data;
    float *c = C->data;

    cuda_gemm_double_buffered(a, b, c, M, N, K);

    // cuda_gemm_naive(a, b, c, M, N, K);
}
