PROJECT = spmv

SRCS = main.cc parboil_opencl.c args.c gpu_info.c file.c convert_dataset.c mmio.c ocl.c

CXXFLAGS += -I.

# Usage: -i matrix_file,vector_file [-o output_file]
OPTS ?= -i 1138_bus.mtx,1138_bus.vec

include ../common.mk
