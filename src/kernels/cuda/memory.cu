#include "cuda_common.cuh"

extern "C" bool pico_cuda_tensor_alloc(struct PicoContext *ctx, struct PicoTensor *tensor) {
    (void)ctx;
    if(tensor == nullptr) {
        return false;
    }

    if(tensor->backend == PICO_BACKEND_CUDA && tensor->data != nullptr && tensor->grad != nullptr) {
        return true;
    }

    if(tensor->backend == PICO_BACKEND_CUDA && (tensor->data != nullptr || tensor->grad != nullptr)) {
        fprintf(stderr, "PicoCudaError: tensor has partial CUDA allocation\n");
        return false;
    }

    if(!pico_cuda_ok(cudaMalloc((void **)&tensor->data, tensor->numel * sizeof(float)), "cudaMalloc data")) {
        return false;
    }
    if(!pico_cuda_ok(cudaMalloc((void **)&tensor->grad, tensor->numel * sizeof(float)), "cudaMalloc grad")) {
        cudaFree(tensor->data);
        tensor->data = nullptr;
        return false;
    }

    cudaMemset(tensor->data, 0, tensor->numel * sizeof(float));
    cudaMemset(tensor->grad, 0, tensor->numel * sizeof(float));
    tensor->backend = PICO_BACKEND_CUDA;
    return true;
}

extern "C" bool pico_cuda_tensor_to_cuda(struct PicoContext *ctx, struct PicoTensor *tensor) {
    if(tensor == nullptr) {
        return false;
    }

    if(tensor->backend == PICO_BACKEND_CUDA) {
        return tensor->data != nullptr && tensor->grad != nullptr;
    }

    float *old_data = tensor->data;
    float *old_grad = tensor->grad;
    int64_t bytes = tensor->numel * sizeof(float);

    tensor->data = nullptr;
    tensor->grad = nullptr;
    if(!pico_cuda_tensor_alloc(ctx, tensor)) {
        tensor->data = old_data;
        tensor->grad = old_grad;
        tensor->backend = PICO_BACKEND_CPU;
        return false;
    }

    if(old_data != nullptr && !pico_cuda_ok(cudaMemcpy(tensor->data, old_data, bytes, cudaMemcpyHostToDevice), "copy data to cuda")) {
        cudaFree(tensor->data);
        cudaFree(tensor->grad);
        tensor->data = old_data;
        tensor->grad = old_grad;
        tensor->backend = PICO_BACKEND_CPU;
        return false;
    }
    if(old_grad != nullptr && !pico_cuda_ok(cudaMemcpy(tensor->grad, old_grad, bytes, cudaMemcpyHostToDevice), "copy grad to cuda")) {
        cudaFree(tensor->data);
        cudaFree(tensor->grad);
        tensor->data = old_data;
        tensor->grad = old_grad;
        tensor->backend = PICO_BACKEND_CPU;
        return false;
    }

    return true;
}

extern "C" bool pico_cuda_tensor_to_cpu(struct PicoContext *ctx, struct PicoTensor *tensor, float *old_data, float *old_grad) {
    (void)ctx;
    if(tensor == nullptr) {
        return false;
    }

    int64_t bytes = tensor->numel * sizeof(float);
    if(old_data != nullptr && tensor->data != nullptr &&
       !pico_cuda_ok(cudaMemcpy(tensor->data, old_data, bytes, cudaMemcpyDeviceToHost), "copy data to cpu")) {
        return false;
    }
    if(old_grad != nullptr && tensor->grad != nullptr &&
       !pico_cuda_ok(cudaMemcpy(tensor->grad, old_grad, bytes, cudaMemcpyDeviceToHost), "copy grad to cpu")) {
        return false;
    }

    cudaFree(old_data);
    cudaFree(old_grad);
    tensor->backend = PICO_BACKEND_CPU;
    return true;
}
