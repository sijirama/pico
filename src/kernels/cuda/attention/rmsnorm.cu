
#include "../cuda_common.cuh"

__device__ static inline float pico_cuda_warp_reduce_sum(float value) {
    for(int offset = 16; offset > 0; offset >>= 1) {
        value += __shfl_down_sync(0xffffffff, value, offset);
    }

    return value;
}

__global__ static void pico_cuda_rmsnorm_kernel(
    const float *__restrict__ input,
    const float *__restrict__ weight,
    float *__restrict__ out,
    int64_t rows,
    int64_t hidden_dim,
    float eps) {
    (void)weight;
    (void)out;
    (void)eps;

    int row = blockIdx.x;
    if(row >= rows) {
        return;
    }

    int64_t base = row * hidden_dim;
    int tid = threadIdx.x;

    extern __shared__ float shared[];

    // NOTE: stride loop to get all values into the 256 shared mem

    float local_sum = 0.0f;
    for(int64_t d = tid; d < hidden_dim; d += blockDim.x) {
        float x = input[base + d];
        local_sum += x * x;
    }

    shared[tid] = local_sum;
    __syncthreads();

    // INFO: mean reduction from 256 values (in shared mem) -> 32 values/ 1 warp

    for(int stride = blockDim.x / 2; stride > 32; stride >>= 1) {
        if(tid < stride) {
            shared[tid] += shared[tid + stride];
        }
        __syncthreads();
    }

    // INFO: warp reduce for the one warp
    if(tid < 32) {
        float sum = shared[tid];
        if(blockDim.x >= 64) {
            sum += shared[tid + 32];
        }

        sum = pico_cuda_warp_reduce_sum(sum);
        if(tid == 0) {
            shared[0] = sum / (float)hidden_dim;
        }
    }

    if(tid == 0) {
        shared[0] = rsqrtf(shared[0] + eps);
    }
    __syncthreads();

    float inv_rms = shared[0];

    // INFO: final loop
    // INFO: block stripe through the entire row and accumulating out

    for(int d = tid; d < hidden_dim; d += blockDim.x) {
        out[base + d] = input[base + d] * inv_rms * weight[d];
    }
}

extern "C" bool pico_cuda_rmsnorm(struct PicoTensor *input, struct PicoTensor *weight, struct PicoTensor *out, float eps) {
    if(!pico_cuda_tensor_ready(input, "rmsnorm", "input") || !pico_cuda_tensor_ready(weight, "rmsnorm", "weight") ||
       !pico_cuda_tensor_ready(out, "rmsnorm", "out")) {
        return false;
    }

    if(input->ndim < 1) {
        fprintf(stderr, "PicoCudaError: rmsnorm input must have at least one dimension\n");
        return false;
    }

    int64_t hidden_dim = input->shape[input->ndim - 1];
    if(hidden_dim <= 0) {
        fprintf(stderr, "PicoCudaError: rmsnorm hidden_dim must be positive\n");
        return false;
    }

    if(weight->numel != hidden_dim) {
        fprintf(stderr, "PicoCudaError: rmsnorm weight numel must match input last dim\n");
        return false;
    }

    if(out->numel != input->numel) {
        fprintf(stderr, "PicoCudaError: rmsnorm output numel must match input numel\n");
        return false;
    }

    int64_t rows = input->numel / hidden_dim;
    int threads = 256;
    dim3 block(threads);                                   // 256 threads per block man
    dim3 grid((unsigned int)rows);                         // block per normalized input row i.e [B,S,D] -> [B*S,D]
    size_t shared_bytes = (size_t)threads * sizeof(float); // one shared float per thread

    pico_cuda_rmsnorm_kernel<<<grid, block, shared_bytes>>>(input->data, weight->data, out->data, rows, hidden_dim, eps);
    return pico_cuda_ok(cudaDeviceSynchronize(), "rmsnorm");
}
