CC = gcc
AR = ar
CFLAGS = -std=c11 -I src -g -Wall -pthread
LDFLAGS = -lm -pthread

SRC_DIR = src
INC_DIR = src
TEST_DIR = tests
OBJ_DIR = obj
LIB_DIR = lib

TARGET = pico
TEST_TARGET = test_pico
ASAN_TARGET = test_pico_asan
STATIC_LIB = $(LIB_DIR)/libpico.a

# AddressSanitizer: detects leaks, use-after-free, double-free, overflows.
# Slower + more memory, so it's a separate dev-only target (never shipped).
ASAN_FLAGS = -fsanitize=address -fno-omit-frame-pointer

# Source files (recursively find all .c files except main.c)
# SRCS = $(filter-out $(SRC_DIR)/main.c, $(wildcard $(SRC_DIR)/*.c))
SRCS = $(filter-out $(SRC_DIR)/main.c, $(shell find $(SRC_DIR) -name '*.c'))
OBJS = $(patsubst $(SRC_DIR)/%.c, $(OBJ_DIR)/%.o, $(SRCS))
MAIN_OBJ = $(OBJ_DIR)/main.o

# Test files (recursively find all .c under tests/, incl. tests/lib/ etc.)
# obj names are flattened by base filename; vpath lets the rule locate the source
# wherever it lives. (test filenames must stay unique across test subdirs.)
TEST_SRCS = $(shell find $(TEST_DIR) -name '*.c')
TEST_OBJS = $(patsubst %, $(OBJ_DIR)/test_%.o, $(basename $(notdir $(TEST_SRCS))))
vpath %.c $(sort $(dir $(TEST_SRCS)))

# Auto-generated header dependency files (one .d per .o, emitted by -MMD -MP).
# Pulled back in via `-include` at the bottom so editing a .h recompiles the .c
# files that include it — no more stale obj/ ghosts.
DEPS = $(OBJS:.o=.d) $(MAIN_OBJ:.o=.d) $(TEST_OBJS:.o=.d)

all: $(TARGET)

lib: $(STATIC_LIB)

$(TARGET): $(OBJS) $(MAIN_OBJ)
	@echo "Linking $@..."
	@$(CC) $^ -o $@ $(LDFLAGS)

$(STATIC_LIB): $(OBJS)
	@echo "Archiving $@..."
	@mkdir -p $(LIB_DIR)
	@$(AR) rcs $@ $^
	@echo "Archive ready: $@"

$(OBJ_DIR)/%.o: $(SRC_DIR)/%.c | $(OBJ_DIR)
	@echo "Compiling $<..."
	@mkdir -p $(dir $@)
	@$(CC) $(CFLAGS) -MMD -MP -c $< -o $@

$(OBJ_DIR)/test_%.o: %.c | $(OBJ_DIR)
	@echo "Compiling test $<..."
	@$(CC) $(CFLAGS) -I $(TEST_DIR) -MMD -MP -c $< -o $@

$(TEST_TARGET): $(OBJS) $(TEST_OBJS)
	@echo "Linking $@..."
	@$(CC) $^ -o $@ $(LDFLAGS)

$(OBJ_DIR):
	@mkdir -p $(OBJ_DIR)

test: $(TEST_TARGET)
	@echo "Running tests..."
	@./$(TEST_TARGET)

run: $(TARGET)
	@./$(TARGET)

# Build the SAME tests with AddressSanitizer on, then run them.
# Compiles sources directly (not via obj/) so it never mixes with the normal build.
asan:
	@echo "Building tests with AddressSanitizer..."
	@$(CC) $(CFLAGS) -I $(TEST_DIR) $(ASAN_FLAGS) $(SRCS) $(TEST_SRCS) -o $(ASAN_TARGET) $(LDFLAGS)
	@echo "Running tests under ASan..."
	@./$(ASAN_TARGET)

# NOTE: benchmarks live in bench/ and are their OWN self-contained env — run them
# from inside bench/ (`cd bench && make <name>`), never from the repo root.

clean:
	@echo "Cleaning..."
	@rm -rf $(OBJ_DIR) $(LIB_DIR) $(TARGET) $(TEST_TARGET) $(ASAN_TARGET) a.out

rebuild: clean all

# pull in the auto-generated header deps (silent if they don't exist yet)
-include $(DEPS)

.PHONY: all lib test run clean rebuild asan
