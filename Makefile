# Simple Makefile for compiling a single CUDA source file (add.cu)

NVCC := nvcc
NVCCFLAGS := -std=c++17 -O2 -Xcompiler -Wall

SRC := add.cu
BIN_DIR := build
BIN := $(BIN_DIR)/add

.PHONY: all clean run

all: $(BIN)

$(BIN): $(SRC)
	@mkdir -p $(BIN_DIR)
	$(NVCC) $(NVCCFLAGS) -o $@ $<

run: $(BIN)
	./$(BIN) $(SEED)

clean:
	rm -rf $(BIN_DIR)
