CC = gcc
CFLAGS = -std=c11 -I src -g -Wall -pthread
LDFLAGS = -lm -pthread

SRC_DIR = src
INC_DIR = src
TEST_DIR = tests
OBJ_DIR = obj

TARGET = pico
TEST_TARGET = test_pico
ASAN_TARGET = test_pico_asan

# AddressSanitizer: detects leaks, use-after-free, double-free, overflows.
# Slower + more memory, so it's a separate dev-only target (never shipped).
ASAN_FLAGS = -fsanitize=address -fno-omit-frame-pointer

# Source files (recursively find all .c files except main.c)
SRCS = $(filter-out $(SRC_DIR)/main.c, $(wildcard $(SRC_DIR)/*.c))
OBJS = $(patsubst $(SRC_DIR)/%.c, $(OBJ_DIR)/%.o, $(SRCS))
MAIN_OBJ = $(OBJ_DIR)/main.o

# Test files (recursively find all .c under tests/, incl. tests/lib/ etc.)
# obj names are flattened by base filename; vpath lets the rule locate the source
# wherever it lives. (test filenames must stay unique across test subdirs.)
TEST_SRCS = $(shell find $(TEST_DIR) -name '*.c')
TEST_OBJS = $(patsubst %, $(OBJ_DIR)/test_%.o, $(basename $(notdir $(TEST_SRCS))))
vpath %.c $(sort $(dir $(TEST_SRCS)))

all: $(TARGET)

$(TARGET): $(OBJS) $(MAIN_OBJ)
	@echo "Linking $@..."
	@$(CC) $^ -o $@ $(LDFLAGS)

$(OBJ_DIR)/%.o: $(SRC_DIR)/%.c | $(OBJ_DIR)
	@echo "Compiling $<..."
	@$(CC) $(CFLAGS) -c $< -o $@

$(OBJ_DIR)/test_%.o: %.c | $(OBJ_DIR)
	@echo "Compiling test $<..."
	@$(CC) $(CFLAGS) -I $(TEST_DIR) -c $< -o $@

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

clean:
	@echo "Cleaning..."
	@rm -rf $(OBJ_DIR) $(TARGET) $(TEST_TARGET) $(ASAN_TARGET) a.out

rebuild: clean all

.PHONY: all test run clean rebuild asan
