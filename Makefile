# Simple Makefile for compiling a single CUDA source file (add.cu)

NVCC := nvcc
NVCCFLAGS := -std=c++17 -O2 -Xcompiler -Wall

SRC := add.cu
BIN_DIR := build
BIN := $(BIN_DIR)/add
REPORT := determ

.PHONY: all clean run

all: $(BIN)

$(BIN): $(SRC)
	@mkdir -p $(BIN_DIR)
	$(NVCC) $(NVCCFLAGS) -o $@ $<

run: $(BIN)
	./$(BIN) $(SEED)

profile: $(BIN)
	nsys profile --output=$(REPORT) ./$(BIN)
	nsys stats --report cuda_gpu_kern_sum $(REPORT).nsys-rep

clean:
	rm -rf $(BIN_DIR)
	rm $(REPORT).sqlite $(REPORT).nsys-rep
