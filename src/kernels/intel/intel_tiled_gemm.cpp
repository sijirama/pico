

#include "../../tensor.h"
#include "device_selector.hpp"
#include <cstddef>
#include <sycl/sycl.hpp>

[[gnu::always_inline]]
inline void intel_tiled_gemm_naive_core(
    sycl::nd_item<2> item,
    float *a,
    float *b,
    sycl::local_accessor<float, 3> local_tile_a,
    sycl::local_accessor<float, 3> local_tile_b,
    int64_t M,
    int64_t K,
    int64_t N,
    size_t TILE_Y,
    size_t TILE_X) {

    int global_row = item.get_global_id(0); // CUDA: blockIdx.y * blockDim.y + threadIdx.y
    int global_col = item.get_global_id(1); // CUDA: blockIdx.x * blockDim.x + threadIdx.x

    int local_row = item.get_local_id(0); // CUDA: threadIdx.y
    int local_col = item.get_local_id(1); // CUDA: threadIdx.x
    

    auto group = item.get_group();

    float *local_ptr_a = &local_tile_a[0][0][0];
    float *global_ptr_a = &a[global_row * K + (0 * TILE_X)];

    float *local_ptr_b = &local_tile_b[0][0][0];
    float *global_ptr_b = &b[(0 * TILE_Y) * N + global_col];

    auto dec_local_ptr_a = sycl::
        address_space_cast<sycl::access::address_space::local_space, sycl::access::decorated::yes>(
            local_ptr_a);
    auto dec_global_ptr_a = sycl::
        address_space_cast<sycl::access::address_space::global_space, sycl::access::decorated::yes>(
            global_ptr_a);

    auto dec_local_ptr_b = sycl::
        address_space_cast<sycl::access::address_space::local_space, sycl::access::decorated::yes>(
            local_ptr_b);
    auto dec_global_ptr_b = sycl::
        address_space_cast<sycl::access::address_space::global_space, sycl::access::decorated::yes>(
            global_ptr_b);

    // Now the member function signature matches perfectly
    sycl::device_event event_a =
        group.async_work_group_copy(dec_local_ptr_a, dec_global_ptr_a, TILE_Y * TILE_X);
    sycl::device_event event_b =
        group.async_work_group_copy(dec_local_ptr_b, dec_global_ptr_b, TILE_Y * TILE_X);

    float sum = 0.0f;

    for(int phase = 0; phase < K / TILE_X; phase++) {

        int nextphase = phase + 1;
        int current_stage = phase % 2;
        int next_stage = (phase + 1) % 2;

        // load into phase 1 here ////////////////////////////

        float *local_ptr_a = &local_tile_a[next_stage][0][0];
        float *global_ptr_a = &a[global_row * K + (nextphase * TILE_X)];

        float *local_ptr_b = &local_tile_b[next_stage][0][0];
        float *global_ptr_b = &b[(nextphase * TILE_Y) * N + global_col];

        auto dec_local_ptr_a = sycl::address_space_cast<
            sycl::access::address_space::local_space,
            sycl::access::decorated::yes>(local_ptr_a);
        auto dec_global_ptr_a = sycl::address_space_cast<
            sycl::access::address_space::global_space,
            sycl::access::decorated::yes>(global_ptr_a);

        auto dec_local_ptr_b = sycl::address_space_cast<
            sycl::access::address_space::local_space,
            sycl::access::decorated::yes>(local_ptr_b);
        auto dec_global_ptr_b = sycl::address_space_cast<
            sycl::access::address_space::global_space,
            sycl::access::decorated::yes>(global_ptr_b);

        // Now the member function signature matches perfectly
        sycl::device_event event_a =
            group.async_work_group_copy(dec_local_ptr_a, dec_global_ptr_a, TILE_Y * TILE_X);
        sycl::device_event event_b =
            group.async_work_group_copy(dec_local_ptr_b, dec_global_ptr_b, TILE_Y * TILE_X);

        ////////////////////////////////////////////////////////////

        //// wait for current load

        // group.wait_for(event_a);
        // group.wait_for(event_b);



        // compute k
    }

    item.barrier(sycl::access::fence_space::local_space); // __syncthreads
}

extern "C" void intel_tiled_gemm_naive(
    sycl::id<2> idx, struct PicoTensor *a, struct PicoTensor *b, struct PicoTensor *out) {

    sycl::queue q(sycl::gpu_selector_v);

    float *device_data_a = sycl::malloc_device<float>(a->numel, q);
    float *device_data_b = sycl::malloc_device<float>(b->numel, q);
    float *device_data_out = sycl::malloc_device<float>(out->numel, q);

    q.memcpy(device_data_a, a->data, sizeof(float) * a->numel);
    q.memcpy(device_data_b, b->data, sizeof(float) * b->numel);

    int64_t M = a->shape[0];
    int64_t K = a->shape[1]; // also b->shape[0]
    int64_t N = b->shape[1];

    size_t TILE_Y = 16;
    size_t TILE_X = 16;

    sycl::range<2> global_grid(M, N);
    sycl::range<2> local_block(TILE_Y, TILE_X);
    sycl::nd_range<2> execution_space(global_grid, local_block);

    q.submit([&](sycl::handler &cgh) {
         sycl::local_accessor<float, 3> shared_tile_a(sycl::range<3>(2, TILE_Y, TILE_X), cgh);
         sycl::local_accessor<float, 3> shared_tile_b(sycl::range<3>(2, TILE_Y, TILE_X), cgh);

         cgh.parallel_for(execution_space, [=](sycl::nd_item<2> item) {
             intel_tiled_gemm_naive_core(
                 item,
                 device_data_a,
                 device_data_b,
                 shared_tile_a,
                 shared_tile_b,
                 M,
                 K,
                 N,
                 TILE_Y,
                 TILE_X);
         });
     }).wait();

    // 5. Copy the computed results back to the host output tensor
    q.memcpy(out->data, device_data_out, sizeof(float) * out->numel).wait();

    // 6. Free the device arrays
    sycl::free(device_data_a, q);
    sycl::free(device_data_b, q);
    sycl::free(device_data_out, q);
}
