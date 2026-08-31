# ==========================================================================
#  pico benchmarks — self-contained env. Run from THIS folder, not the root.
#
#    make matmul     build + run ONE benchmark (bench_matmul.c -> `matmul`)
#    make all        build + run every benchmark
#    make clean      remove built binaries
#
#  Benchmarks compile at -O2 (a -g/-O0 build is meaningless for timing) and link
#  the pico library sources from ../src directly. Add a new top-level benchmark
#  by dropping bench_<name>.c here, or a matmul benchmark in matmul/bench_<name>.c.
# ==========================================================================

CC       = gcc
SRC_DIR  = ../src
MATMUL_DIR = matmul
CFLAGS   = -std=c11 -O2 -I $(SRC_DIR) -Wall -pthread -fopenmp
LDFLAGS  = -lm -pthread -fopenmp
CJSON_LIBS = -lcjson
BIN      = bin

# pico library sources (everything except the app entrypoint)
SRCS = $(filter-out $(SRC_DIR)/main.c, $(shell find $(SRC_DIR) -name '*.c'))

# one target per bench_<name>.c   ->   `make <name>`
ROOT_BENCHES = $(patsubst bench_%.c,%,$(wildcard bench_*.c))
ROOT_BENCHES := $(filter-out blas_compare,$(ROOT_BENCHES))
MATMUL_BENCHES = $(patsubst $(MATMUL_DIR)/bench_%.c,%,$(wildcard $(MATMUL_DIR)/bench_*.c))
MATMUL_BENCHES := $(filter-out matmul_mkl,$(MATMUL_BENCHES))
BENCHES = $(ROOT_BENCHES) $(MATMUL_BENCHES)

.PHONY: all clean $(BENCHES) blas_openblas blas_blis matmul_focused_openblas matmul_mkl

all: $(BENCHES)

# .PHONY so every invocation rebuilds + reruns with a fresh binary
$(ROOT_BENCHES): %: bench_%.c $(SRCS) | $(BIN)
	@echo ">> building $@  (-O2)"
	@$(CC) $(CFLAGS) $< $(SRCS) -o $(BIN)/$@ $(LDFLAGS) $(CJSON_LIBS)
	@echo ">> running  $@"
	@./$(BIN)/$@

$(MATMUL_BENCHES): %: $(MATMUL_DIR)/bench_%.c $(SRCS) | $(BIN)
	@echo ">> building $@  (-O2)"
	@$(CC) $(CFLAGS) $< $(SRCS) -o $(BIN)/$@ $(LDFLAGS) $(CJSON_LIBS)
	@echo ">> running  $@"
	@./$(BIN)/$@

blas_openblas: bench_blas_compare.c $(SRCS) | $(BIN)
	@echo ">> building $@  (-O2, OpenBLAS)"
	@$(CC) $(CFLAGS) -DBLAS_NAME=\"OpenBLAS\" $< $(SRCS) -o $(BIN)/$@ $(LDFLAGS) $(CJSON_LIBS) -lopenblas
	@echo ">> running  $@"
	@./$(BIN)/$@

blas_blis: bench_blas_compare.c $(SRCS) | $(BIN)
	@echo ">> building $@  (-O2, BLIS)"
	@$(CC) $(CFLAGS) -DBLAS_NAME=\"BLIS\" $< $(SRCS) -o $(BIN)/$@ $(LDFLAGS) $(CJSON_LIBS) -lblis
	@echo ">> running  $@"
	@./$(BIN)/$@

matmul_focused_openblas: $(MATMUL_DIR)/bench_matmul_focused.c $(SRCS) | $(BIN)
	@echo ">> building $@  (-O2, OpenBLAS)"
	@$(CC) $(CFLAGS) -DUSE_CBLAS -DBLAS_NAME=\"OpenBLAS\" $< $(SRCS) -o $(BIN)/$@ $(LDFLAGS) $(CJSON_LIBS) -lopenblas
	@echo ">> running  $@"
	@./$(BIN)/$@

matmul_mkl: $(MATMUL_DIR)/bench_matmul_mkl.c $(MATMUL_DIR)/bench_matmul_focused.c $(SRCS) | $(BIN)
	@if [ -z "$(MKLROOT)" ]; then \
		echo "MKLROOT is not set. Run: source /opt/intel/oneapi/setvars.sh"; \
		exit 1; \
	fi
	@echo ">> building $@  (-O2, Intel MKL)"
	@$(CC) $(CFLAGS) -I$(MKLROOT)/include $< $(SRCS) -o $(BIN)/$@ $(LDFLAGS) $(CJSON_LIBS) -L$(MKLROOT)/lib/intel64 -Wl,-rpath,$(MKLROOT)/lib/intel64 -lmkl_rt
	@echo ">> running  $@"
	@./$(BIN)/$@

$(BIN):
	@mkdir -p $(BIN)

clean:
	@echo "cleaning bench binaries + assembly dumps..."
	@rm -rf $(BIN) *.s
