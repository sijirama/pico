#pragma once

void cuda_gemm_naive(
    const float *A,
    const float *B,
    float *C,
    int M,
    int N,
    int K,
    int batch_count);

void cuda_gemm_tiled(
    const float *A,
    const float *B,
    float *C,
    int M,
    int N,
    int K,
    int batch_count);

void cuda_gemm_double_buffered(
    const float *A,
    const float *B,
    float *C,
    int M,
    int N,
    int K,
    int batch_count);
