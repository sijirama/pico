#include "pico.h"

#include <stdio.h>

void print_welcome() {
    printf("Welcome to pico\n");
}

void print_version() {
    printf("pico version %s\n", PICO_VERSION);
}
