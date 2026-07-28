struct PicoContext;

// ========== SGD

struct PicoOptimSGD {
    float lr;
};

struct PicoOptimSGD* pico_optim_sgd_init(float lr);
void pico_optim_sgd_step(struct PicoContext* ctx, struct PicoOptimSGD* optim);
void pico_optim_sgd_zero_grad(struct PicoContext* ctx, struct PicoOptimSGD* optim);
void pico_optim_sgd_free(struct PicoOptimSGD* optim);


// ==================== Nesterov accelerated gradient (NAG)
// ==================== AdaGrad
// ==================== RMSProp
// ==================== ADAM
// ==================== Muon
