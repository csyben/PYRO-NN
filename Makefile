TF_CFLAGS := $(shell python -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_compile_flags()))')
TF_LFLAGS := $(shell python -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_link_flags()))')
CUDA_PATH := /usr/local/cuda/targets/x86_64-linux
NVCC := nvcc
GCC := g++
NVCC_FLAGS := -std=c++14 -c -D GOOGLE_CUDA=1 -x cu -Xcompiler -fPIC --expt-relaxed-constexpr
GCC_FLAGS := -std=c++14 -shared -fPIC

# Define build directories
BUILD_DIR := build
BUILD_KERNELS_DIR := $(BUILD_DIR)/tensorflow_kernels
BUILD_OPS_DIR := $(BUILD_DIR)/tensorflow_ops

# Create build directories
$(shell mkdir -p $(BUILD_KERNELS_DIR))
$(shell mkdir -p $(BUILD_OPS_DIR))

# Define source and target files
SRC := src/pyronn/ct_reconstruction/cpp
KERNEL_SRC := $(wildcard $(SRC)/tensorflow_kernels/*.cu.cc)
KERNEL_OBJS := $(patsubst $(SRC)/tensorflow_kernels/%.cu.cc, $(BUILD_KERNELS_DIR)/%.cu.o, $(KERNEL_SRC))

OPS_SRC := $(wildcard $(SRC)/tensorflow_ops/*.cc)

# Default target
all: src/pyronn_layers/pyronn_layers_tensorflow.so

# Generic rules for building kernels
$(BUILD_KERNELS_DIR)/%.cu.o: $(SRC)/tensorflow_kernels/%.cu.cc
	$(NVCC) $(NVCC_FLAGS) -o $@ $< $(TF_CFLAGS)

# Rules for building ops
src/pyronn_layers/pyronn_layers_tensorflow.so: $(KERNEL_OBJS) $(OPS_SRC)
	$(GCC) $(GCC_FLAGS) -o $@ $^ $(TF_CFLAGS) $(TF_LFLAGS) -I$(CUDA_PATH)/include -L$(CUDA_PATH)/lib  -lcudart
#
#check:
#	@echo "Files in current directory:"
#	@ls -1
# Clean
clean:
	rm -rf $(BUILD_DIR)

.PHONY: all clean