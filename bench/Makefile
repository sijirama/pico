# ==========================================================================
#  pico benchmarks — self-contained env. Run from THIS folder, not the root.
#
#    make matmul     build + run ONE benchmark (bench_matmul.c -> `matmul`)
#    make all        build + run every benchmark
#    make clean      remove built binaries
#
#  Benchmarks compile at -O2 (a -g/-O0 build is meaningless for timing) and link
#  the pico library sources from ../src directly. Add a new benchmark by dropping
#  bench_<name>.c (with its own main()) here — it auto-becomes `make <name>`.
# ==========================================================================

CC       = gcc
SRC_DIR  = ../src
CFLAGS   = -std=c11 -O2 -I $(SRC_DIR) -Wall -pthread
LDFLAGS  = -lm -pthread
BIN      = bin

# pico library sources (everything except the app entrypoint)
SRCS = $(filter-out $(SRC_DIR)/main.c, $(shell find $(SRC_DIR) -name '*.c'))

# one target per bench_<name>.c   ->   `make <name>`
BENCHES = $(patsubst bench_%.c,%,$(wildcard bench_*.c))
BENCHES := $(filter-out blas_compare,$(BENCHES))

.PHONY: all clean $(BENCHES) blas_openblas blas_blis

all: $(BENCHES)

# .PHONY so every invocation rebuilds + reruns with a fresh binary
$(BENCHES): %: bench_%.c $(SRCS) | $(BIN)
	@echo ">> building $@  (-O2)"
	@$(CC) $(CFLAGS) $< $(SRCS) -o $(BIN)/$@ $(LDFLAGS)
	@echo ">> running  $@"
	@./$(BIN)/$@

blas_openblas: bench_blas_compare.c $(SRCS) | $(BIN)
	@echo ">> building $@  (-O2, OpenBLAS)"
	@$(CC) $(CFLAGS) -DBLAS_NAME=\"OpenBLAS\" $< $(SRCS) -o $(BIN)/$@ $(LDFLAGS) -lopenblas
	@echo ">> running  $@"
	@./$(BIN)/$@

blas_blis: bench_blas_compare.c $(SRCS) | $(BIN)
	@echo ">> building $@  (-O2, BLIS)"
	@$(CC) $(CFLAGS) -DBLAS_NAME=\"BLIS\" $< $(SRCS) -o $(BIN)/$@ $(LDFLAGS) -lblis
	@echo ">> running  $@"
	@./$(BIN)/$@

$(BIN):
	@mkdir -p $(BIN)

clean:
	@echo "cleaning bench binaries + assembly dumps..."
	@rm -rf $(BIN) *.s
