# Simple Makefile for compiling a single CUDA source file (add.cu)

NVCC := nvcc
NVCCFLAGS := -std=c++17 -O2 -Xcompiler -Wall

SRC := add.cu
BIN_DIR := build
BIN := $(BIN_DIR)/add
REPORT := determ

# Optional runtime args: pass -s <seed> and/or -i <iterations> only when provided
ARGS := $(if $(SEED),-s $(SEED)) $(if $(ITERS),-i $(ITERS))

.PHONY: all clean run profile help


all: $(BIN)

$(BIN): $(SRC)
	@mkdir -p $(BIN_DIR)
	$(NVCC) $(NVCCFLAGS) -o $@ $<

run: $(BIN)
	./$(BIN) $(ARGS)

profile: $(BIN)
	nsys profile --output=$(REPORT) ./$(BIN) $(ARGS)
	nsys stats --report cuda_gpu_kern_sum $(REPORT).nsys-rep

clean:
	rm -rf $(BIN_DIR)
	rm $(REPORT).sqlite $(REPORT).nsys-rep

# Help target: explains how to run the demo and use optional SEED/ITERS variables
help:
	@echo "Usage: make [target] [SEED=<seed>] [ITERS=<n>]"
	@echo
	@echo "Targets:"
	@echo "  all       - build the binary"
	@echo "  run       - run the demo (passes -s/--seed and -i/--iter to the program if SEED/ITERS are set)"
	@echo "             Examples:"
	@echo "               make run"
	@echo "               make run SEED=42"
	@echo "               make run ITERS=200"
	@echo "               make run SEED=42 ITERS=200"
	@echo "  profile   - profile with nsys (uses same SEED/ITERS mapping)"
	@echo "  clean     - remove build artifacts"
	@echo
