
#include "device_selector.hpp"
#include "iintel_gemm.h"
#include <sycl/sycl.hpp>

[[gnu::always_inline]]
inline void scaling_kernel_core(sycl::id<1> idx, float *device_data, float factor) {
    device_data[idx] *= factor;
}

// 2. The Host Launcher Bridge
extern "C" void scale_array_gpu(float *data, int size, float factor) {
    sycl::queue q(sycl::gpu_selector_v);

    // Allocate Intel iGPU USM memory
    float *device_data = sycl::malloc_device<float>(size, q);

    // Copy to iGPU
    q.memcpy(device_data, data, sizeof(float) * size).wait();

    q.parallel_for(sycl::range<1>(size), [=](sycl::id<1> idx) {
         scaling_kernel_core(idx, device_data, factor);
     }).wait();

    // Copy results back
    q.memcpy(data, device_data, sizeof(float) * size).wait();

    // Cleanup
    sycl::free(device_data, q);
}
