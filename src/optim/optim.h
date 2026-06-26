
#include "lib/pico_vector.h"
#include "tensor.h"

// ========== SGD

struct PicoOptimSGD {
    struct PicoVec params;
    float lr;
};

struct PicoOptimSGD* pico_optim_sgd_init(float lr);
void pico_optim_sgd_add(struct PicoOptimSGD* optim, struct PicoTensor* param);
void pico_optim_sgd_step(struct PicoOptimSGD* optim);
void pico_optim_sgd_zero_grad(struct PicoOptimSGD* optim);
void pico_optim_sgd_free(struct PicoOptimSGD* optim);


// ==================== Nesterov accelerated gradient (NAG)
// ==================== AdaGrad
// ==================== RMSProp
// ==================== ADAM
// ==================== Muon
