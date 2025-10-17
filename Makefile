# Simple Makefile for compiling a single CUDA source file (add.cu)

NVCC := nvcc
NVCCFLAGS := -std=c++14 -O2 -Xcompiler -Wall

SRC := add.cu
BIN := add

.PHONY: all clean run

all: $(BIN)

$(BIN): $(SRC)
	$(NVCC) $(NVCCFLAGS) -o $@ $<

run: $(BIN)
	./$(BIN)

clean:
	rm -f $(BIN)
