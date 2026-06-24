#pragma once

struct PicoTensor* pico_add(struct PicoTensor* a, struct PicoTensor* b);
struct PicoTensor* pico_sub(struct PicoTensor* a, struct PicoTensor* b);
struct PicoTensor* pico_matmul(struct PicoTensor* a, struct PicoTensor* b);
