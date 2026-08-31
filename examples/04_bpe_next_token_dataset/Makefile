CC = gcc
CFLAGS = -std=c11 -Wall -I ../../src -fopenmp
LDFLAGS = -L ../../lib -lpico -lm -pthread -fopenmp
CJSON_LIBS = -lcjson

TARGET = 04_bpe_next_token_dataset

all: $(TARGET)

pico-lib:
	$(MAKE) -C ../.. lib
	@echo "pico static library is ready."

$(TARGET): main.c pico-lib
	$(CC) $(CFLAGS) main.c -o $(TARGET) $(LDFLAGS) $(CJSON_LIBS)

run: $(TARGET)
	./$(TARGET)

clean:
	rm -f $(TARGET)

.PHONY: all pico-lib run clean
