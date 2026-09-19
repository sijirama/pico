
#include "cuda_backend.h"
#include "gemm/gemm.cuh"

extern "C" void pico_cuda_matmul(PicoTensor *A, PicoTensor *B, PicoTensor *C) {

    int M = A->shape[A->ndim - 2];
    int K = A->shape[A->ndim - 1];
    int N = B->shape[B->ndim - 1];

    int batch_count = 1;
    for(int i = 0; i < C->ndim - 2; i++) {
        batch_count *= C->shape[i];
    }

    float *a_host = A->data;
    float *b_host = B->data;
    float *c_host = C->data;

    float *a_device;
    float *b_device;
    float *c_device;

    // Allocate device memory
    cudaMalloc((void **)&a_device, batch_count * M * K * sizeof(float));
    cudaMalloc((void **)&b_device, batch_count * K * N * sizeof(float));
    cudaMalloc((void **)&c_device, batch_count * M * N * sizeof(float));

    // Copy data from host to device
    cudaMemcpy(
        a_device,
        a_host,
        batch_count * M * K * sizeof(float),
        cudaMemcpyHostToDevice);
    cudaMemcpy(
        b_device,
        b_host,
        batch_count * K * N * sizeof(float),
        cudaMemcpyHostToDevice);

    // Perform GEMM on device
    cuda_gemm_naive(a_device, b_device, c_device, M, N, K, batch_count);

    // Copy result back to host
    cudaMemcpy(
        c_host,
        c_device,
        batch_count * M * N * sizeof(float),
        cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(a_device);
    cudaFree(b_device);
    cudaFree(c_device);
}
