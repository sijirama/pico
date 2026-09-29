struct PicoContext;
struct PicoTensor;

// ========== SGD

struct PicoOptimSGD {
    float lr;
};

struct PicoOptimSGD* pico_optim_sgd_init(float lr);
void pico_optim_sgd_step(struct PicoContext* ctx, struct PicoOptimSGD* optim);
void pico_optim_sgd_zero_grad(struct PicoContext* ctx, struct PicoOptimSGD* optim);
void pico_optim_sgd_free(struct PicoOptimSGD* optim);


// ==================== ADAM

struct PicoOptimAdam {
    float lr;
    float beta1;
    float beta2;
    float eps;
    int step;
    int param_count;
    struct PicoTensor** params;
    float** m;
    float** v;
};

struct PicoOptimAdam* pico_optim_adam_init(float lr);
void pico_optim_adam_step(struct PicoContext* ctx, struct PicoOptimAdam* optim);
void pico_optim_adam_zero_grad(struct PicoContext* ctx, struct PicoOptimAdam* optim);
void pico_optim_adam_free(struct PicoOptimAdam* optim);


// ==================== ADAMW

struct PicoOptimAdamW {
    float lr;
    float beta1;
    float beta2;
    float eps;
    float weight_decay;
    int step;
    int param_count;
    struct PicoTensor** params;
    float** m;
    float** v;
};

struct PicoOptimAdamW* pico_optim_adamw_init(float lr, float weight_decay);
void pico_optim_adamw_step(struct PicoContext* ctx, struct PicoOptimAdamW* optim);
void pico_optim_adamw_zero_grad(struct PicoContext* ctx, struct PicoOptimAdamW* optim);
void pico_optim_adamw_free(struct PicoOptimAdamW* optim);


// ==================== Nesterov accelerated gradient (NAG)
// ==================== AdaGrad
// ==================== RMSProp
// ==================== Muon
