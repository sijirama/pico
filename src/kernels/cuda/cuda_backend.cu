
#include "cuda_backend.h"
#include "gemm/gemm.cuh"

extern "C" void pico_cuda_matmul(PicoTensor *A, PicoTensor *B, PicoTensor *C) {

    int M = A->shape[0];
    int K = A->shape[1];
    int N = B->shape[1];

    float *a_host = A->data;
    float *b_host = B->data;
    float *c_host = C->data;

    float *a_device;
    float *b_device;
    float *c_device;

    // Allocate device memory
    cudaMalloc((void **)&a_device, M * K * sizeof(float));
    cudaMalloc((void **)&b_device, K * N * sizeof(float));
    cudaMalloc((void **)&c_device, M * N * sizeof(float));

    // Copy data from host to device
    cudaMemcpy(a_device, a_host, M * K * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(b_device, b_host, K * N * sizeof(float), cudaMemcpyHostToDevice);

    // Perform GEMM on device
    cuda_gemm_double_buffered(a_device, b_device, c_device, M, N, K);

    // Copy result back to host
    cudaMemcpy(c_host, c_device, M * N * sizeof(float), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(a_device);
    cudaFree(b_device);
    cudaFree(c_device);
}
