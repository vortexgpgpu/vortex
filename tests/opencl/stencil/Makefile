PROJECT = stencil

SRCS = main.cc args.c parboil_opencl.c ocl.c gpu_info.c file.c

CXXFLAGS += -I.

# Usage: #nx #ny #nz #iter -i input_file [-o output_file]
OPTS ?= 64 64 8 1 -i 64x64x8.bin

include ../common.mk
