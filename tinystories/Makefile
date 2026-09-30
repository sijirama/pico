CC = gcc
CFLAGS = -std=c11 -Wall -g -pthread -fopenmp -I ../src -I .
LDFLAGS = -lm -pthread -fopenmp -lcjson

SRC = $(filter-out ../src/main.c, $(shell find ../src -name '*.c'))
TARGET = tinystories_cpu_smoke

all: $(TARGET)

$(TARGET): train.c dataset.c main.h
	$(CC) $(CFLAGS) train.c dataset.c $(SRC) -o $(TARGET) $(LDFLAGS)

run: $(TARGET)
	cd .. && ./tinystories/$(TARGET)

clean:
	rm -f $(TARGET)

.PHONY: all run clean
