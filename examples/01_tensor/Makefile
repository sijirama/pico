CC = gcc
CFLAGS = -std=c11 -Wall -I ../../src
LDFLAGS = -L ../../lib -lpico -lm -pthread

TARGET = 01_tensor

all: $(TARGET)

pico-lib:
	$(MAKE) -C ../.. lib
	@echo "pico static library is ready."

$(TARGET): main.c pico-lib
	$(CC) $(CFLAGS) main.c -o $(TARGET) $(LDFLAGS)

run: $(TARGET)
	./$(TARGET)

clean:
	rm -f $(TARGET)

.PHONY: all pico-lib run clean
