#pragma once

#include <stdbool.h>

struct PicoTensor;
struct PicoContext;

typedef enum { PICO_BACKEND_CPU, PICO_BACKEND_CUDA } PicoBackend;

const char* pico_backend_name(PicoBackend backend);
bool pico_require_same_backend(struct PicoTensor* a, struct PicoTensor* b, const char* op_name);
bool pico_require_cpu_backend(PicoBackend backend, const char* op_name);
bool pico_tensor_init_data_on(struct PicoContext* ctx, struct PicoTensor* tensor, PicoBackend backend);
bool pico_tensor_to_backend(struct PicoContext* ctx, struct PicoTensor* tensor, PicoBackend backend);
