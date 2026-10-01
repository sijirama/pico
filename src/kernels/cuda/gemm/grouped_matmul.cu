#include "../cuda_common.cuh"

#define PICO_CUDA_GEMM_TILE 16

__global__ static void pico_cuda_grouped_matmul_kernel(const float* __restrict__ A,
                                                       const float* __restrict__ B,
                                                       float* __restrict__ C,
                                                       int batch_count,
                                                       int hq,
                                                       int hkv,
                                                       int M,
                                                       int N,
                                                       int K,
                                                       int group_size) {
    int local_row = threadIdx.y;
    int local_col = threadIdx.x;
    int row = blockIdx.y * blockDim.y + local_row;
    int col = blockIdx.x * blockDim.x + local_col;
    int z = blockIdx.z;

    int batch = z / hq;
    int q_head = z % hq;
    int kv_head = q_head / group_size;

    if(batch >= batch_count || q_head >= hq || kv_head >= hkv) {
        return;
    }

    const float* __restrict__ a_current = A + ((batch * hq + q_head) * M * K);
    const float* __restrict__ b_current = B + ((batch * hkv + kv_head) * K * N);
    float* __restrict__ c_current = C + ((batch * hq + q_head) * M * N);

    __shared__ float a_tile[PICO_CUDA_GEMM_TILE][PICO_CUDA_GEMM_TILE];
    __shared__ float b_tile[PICO_CUDA_GEMM_TILE][PICO_CUDA_GEMM_TILE];

    float sum = 0.0f;
    for(int tile = 0; tile < K; tile += PICO_CUDA_GEMM_TILE) {
        int a_col = tile + local_col;
        int b_row = tile + local_row;

        a_tile[local_row][local_col] = (row < M && a_col < K) ? a_current[row * K + a_col] : 0.0f;
        b_tile[local_row][local_col] = (b_row < K && col < N) ? b_current[b_row * N + col] : 0.0f;

        __syncthreads();

        for(int k = 0; k < PICO_CUDA_GEMM_TILE; k++) {
            sum += a_tile[local_row][k] * b_tile[k][local_col];
        }

        __syncthreads();
    }

    if(row < M && col < N) {
        c_current[row * N + col] = sum;
    }
}

extern "C" bool pico_cuda_grouped_matmul(struct PicoTensor* a,
                                         struct PicoTensor* b,
                                         struct PicoTensor* out,
                                         int group_size) {
    if(!pico_cuda_tensor_ready(a, "grouped_matmul", "a") ||
       !pico_cuda_tensor_ready(b, "grouped_matmul", "b") ||
       !pico_cuda_tensor_ready(out, "grouped_matmul", "out")) {
        return false;
    }

    if(a->ndim != 4 || b->ndim != 4 || out->ndim != 4) {
        fprintf(stderr, "PicoCudaError: grouped_matmul only supports 4D tensors\n");
        return false;
    }

    if(group_size <= 0) {
        fprintf(stderr, "PicoCudaError: grouped_matmul group_size must be positive\n");
        return false;
    }

    int batch_count = (int)a->shape[0];
    int hq = (int)a->shape[1];
    int hkv = (int)b->shape[1];
    int M = (int)a->shape[2];
    int K = (int)a->shape[3];
    int N = (int)b->shape[3];

    if(a->shape[0] != b->shape[0] || a->shape[3] != b->shape[2] || hq != hkv * group_size) {
        fprintf(stderr, "PicoCudaError: grouped_matmul received incompatible shapes\n");
        return false;
    }

    if(out->shape[0] != a->shape[0] || out->shape[1] != a->shape[1] || out->shape[2] != a->shape[2] ||
       out->shape[3] != b->shape[3]) {
        fprintf(stderr, "PicoCudaError: grouped_matmul output shape is incorrect\n");
        return false;
    }

    dim3 block(PICO_CUDA_GEMM_TILE, PICO_CUDA_GEMM_TILE);
    dim3 grid((N + block.x - 1) / block.x, (M + block.y - 1) / block.y, batch_count * hq);

    pico_cuda_grouped_matmul_kernel<<<grid, block>>>(a->data, b->data, out->data, batch_count, hq, hkv, M, N, K,
                                                     group_size);
    return pico_cuda_ok(cudaDeviceSynchronize(), "grouped_matmul");
}

