#pragma once

#include "../../tensor.h"

#ifdef __cplusplus
extern "C" {
#endif

//////////////////////////////



void pico_cuda_matmul(
    struct PicoTensor *A,
    struct PicoTensor *B,
    struct PicoTensor *C
);



















/////////////////////
#ifdef __cplusplus
}
#endif
