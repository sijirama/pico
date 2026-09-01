CC = gcc
CFLAGS = -std=c11 -Wall -I ../../src -fopenmp
LDFLAGS = -L ../../lib -lpico -lm -pthread -fopenmp
CJSON_LIBS = -lcjson

TARGET = 06_train_mlp_save_safetensor

all: $(TARGET)

pico-lib:
	$(MAKE) -C ../.. lib
	@echo "pico static library is ready."

$(TARGET): main.c pico-lib
	$(CC) $(CFLAGS) main.c -o $(TARGET) $(LDFLAGS) $(CJSON_LIBS)

run: $(TARGET)
	./$(TARGET)

clean:
	rm -f $(TARGET) trained_relu_mlp.safetensors

.PHONY: all pico-lib run clean
