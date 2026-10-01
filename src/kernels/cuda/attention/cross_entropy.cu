#include "../cuda_common.cuh"

__device__ static inline float pico_cuda_block_reduce_sum(float value, float* shared) {
    int tid = threadIdx.x;
    shared[tid] = value;
    __syncthreads();

    for(int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if(tid < stride) {
            shared[tid] += shared[tid + stride];
        }
        __syncthreads();
    }

    return shared[0];
}

__device__ static inline float pico_cuda_block_reduce_max(float value, float* shared) {
    int tid = threadIdx.x;
    shared[tid] = value;
    __syncthreads();

    for(int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if(tid < stride && shared[tid + stride] > shared[tid]) {
            shared[tid] = shared[tid + stride];
        }
        __syncthreads();
    }

    return shared[0];
}

__global__ static void pico_cuda_cross_entropy_rows_kernel(const float* __restrict__ logits,
                                                           const float* __restrict__ targets,
                                                           float* __restrict__ row_losses,
                                                           int64_t rows,
                                                           int64_t classes,
                                                           int* __restrict__ invalid_target) {
    int64_t row = blockIdx.x;
    int tid = threadIdx.x;

    if(row >= rows) {
        return;
    }

    extern __shared__ float shared[];
    int64_t base = row * classes;

    float local_max = -INFINITY;
    for(int64_t c = tid; c < classes; c += blockDim.x) {
        float value = logits[base + c];
        if(value > local_max) {
            local_max = value;
        }
    }

    float max_logit = pico_cuda_block_reduce_max(local_max, shared);

    float local_sum_exp = 0.0f;
    for(int64_t c = tid; c < classes; c += blockDim.x) {
        local_sum_exp += expf(logits[base + c] - max_logit);
    }

    float sum_exp = pico_cuda_block_reduce_sum(local_sum_exp, shared);

    if(tid == 0) {
        int64_t target = (int64_t)targets[row];
        if(target < 0 || target >= classes || targets[row] != (float)target) {
            *invalid_target = 1;
            row_losses[row] = 0.0f;
            return;
        }

        row_losses[row] = max_logit + logf(sum_exp) - logits[base + target];
    }
}

__global__ static void pico_cuda_cross_entropy_reduce_kernel(const float* __restrict__ row_losses,
                                                            float* __restrict__ out,
                                                            int64_t rows,
                                                            int reduction) {
    int tid = threadIdx.x;
    extern __shared__ float shared[];

    float local_sum = 0.0f;
    for(int64_t i = tid; i < rows; i += blockDim.x) {
        local_sum += row_losses[i];
    }

    float loss = pico_cuda_block_reduce_sum(local_sum, shared);
    if(tid == 0) {
        out[0] = reduction == 0 ? loss / (float)rows : loss;
    }
}

extern "C" bool pico_cuda_cross_entropy(struct PicoTensor* logits,
                                        struct PicoTensor* targets,
                                        struct PicoTensor* out,
                                        int reduction) {
    if(!pico_cuda_tensor_ready(logits, "cross_entropy", "logits") ||
       !pico_cuda_tensor_ready(targets, "cross_entropy", "targets") ||
       !pico_cuda_tensor_ready(out, "cross_entropy", "out")) {
        return false;
    }

    if(logits->ndim < 1) {
        fprintf(stderr, "PicoCudaError: cross_entropy logits must have at least one dimension\n");
        return false;
    }

    int64_t classes = logits->shape[logits->ndim - 1];
    if(classes <= 0 || logits->numel % classes != 0) {
        fprintf(stderr, "PicoCudaError: cross_entropy logits last dimension is invalid\n");
        return false;
    }

    int64_t rows = logits->numel / classes;
    if(targets->numel != rows || out->numel != 1) {
        fprintf(stderr, "PicoCudaError: cross_entropy target/output shape is invalid\n");
        return false;
    }

    float* row_losses = nullptr;
    int* invalid_target = nullptr;

    if(!pico_cuda_ok(cudaMalloc((void**)&row_losses, (size_t)rows * sizeof(float)), "cross_entropy row loss allocation")) {
        return false;
    }

    if(!pico_cuda_ok(cudaMalloc((void**)&invalid_target, sizeof(int)), "cross_entropy target flag allocation")) {
        cudaFree(row_losses);
        return false;
    }

    if(!pico_cuda_ok(cudaMemset(invalid_target, 0, sizeof(int)), "cross_entropy target flag reset")) {
        cudaFree(invalid_target);
        cudaFree(row_losses);
        return false;
    }

    int threads = 256;
    size_t shared_bytes = (size_t)threads * sizeof(float);

    pico_cuda_cross_entropy_rows_kernel<<<(unsigned int)rows, threads, shared_bytes>>>(
        logits->data, targets->data, row_losses, rows, classes, invalid_target);
    if(!pico_cuda_ok(cudaGetLastError(), "cross_entropy rows kernel")) {
        cudaFree(invalid_target);
        cudaFree(row_losses);
        return false;
    }

    int host_invalid_target = 0;
    if(!pico_cuda_ok(cudaMemcpy(&host_invalid_target, invalid_target, sizeof(int), cudaMemcpyDeviceToHost),
                     "cross_entropy target flag copy")) {
        cudaFree(invalid_target);
        cudaFree(row_losses);
        return false;
    }

    if(host_invalid_target != 0) {
        fprintf(stderr, "PicoCudaError: cross_entropy targets must be integer class ids in range\n");
        cudaFree(invalid_target);
        cudaFree(row_losses);
        return false;
    }

    pico_cuda_cross_entropy_reduce_kernel<<<1, threads, shared_bytes>>>(row_losses, out->data, rows, reduction);
    bool ok = pico_cuda_ok(cudaDeviceSynchronize(), "cross_entropy");

    cudaFree(invalid_target);
    cudaFree(row_losses);
    return ok;
}

