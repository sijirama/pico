CC = gcc
CFLAGS = -std=c11 -I src -g -Wall
LDFLAGS = -lm

SRC_DIR = src
INC_DIR = src
TEST_DIR = tests
OBJ_DIR = obj

TARGET = pico
TEST_TARGET = test_pico

# Source files (recursively find all .c files except main.c)
SRCS = $(filter-out $(SRC_DIR)/main.c, $(wildcard $(SRC_DIR)/*.c))
OBJS = $(patsubst $(SRC_DIR)/%.c, $(OBJ_DIR)/%.o, $(SRCS))
MAIN_OBJ = $(OBJ_DIR)/main.o

# Test files
TEST_SRCS = $(wildcard $(TEST_DIR)/*.c)
TEST_OBJS = $(patsubst $(TEST_DIR)/%.c, $(OBJ_DIR)/test_%.o, $(TEST_SRCS))

all: $(TARGET)

$(TARGET): $(OBJS) $(MAIN_OBJ)
	@echo "Linking $@..."
	@$(CC) $^ -o $@ $(LDFLAGS)

$(OBJ_DIR)/%.o: $(SRC_DIR)/%.c | $(OBJ_DIR)
	@echo "Compiling $<..."
	@$(CC) $(CFLAGS) -c $< -o $@

$(OBJ_DIR)/test_%.o: $(TEST_DIR)/%.c | $(OBJ_DIR)
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

clean:
	@echo "Cleaning..."
	@rm -rf $(OBJ_DIR) $(TARGET) $(TEST_TARGET) a.out 

rebuild: clean all

.PHONY: all test run clean rebuild
