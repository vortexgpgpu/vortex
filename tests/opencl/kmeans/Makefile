ROOT_DIR := $(realpath ../../..)
include $(ROOT_DIR)/config.mk

PROJECT := kmeans

SRC_DIR := $(VORTEX_HOME)/tests/opencl/$(PROJECT)

SRCS := $(SRC_DIR)/main.cc $(SRC_DIR)/read_input.c $(SRC_DIR)/rmse.c $(SRC_DIR)/kmeans_clustering.c $(SRC_DIR)/cluster.c $(SRC_DIR)/getopt.c

kernel.cl: $(SRC_DIR)/kernel.cl
	cp $< $@

KERNEL_SRCS := kernel.cl

# Self-contained small case (deterministic synthetic blobs, no input file).
# usage: -f<features> -p<points>  (p must keep global work a multiple-friendly size)
OPTS ?= -f16 -p512

# No atomics required: the GPU kernel is assignment-only (nearest-centroid);
# centroid recomputation is done host-side. (If the update ever moves on-device
# with atomic accumulation, add CONFIGS=-DVX_CFG_EXT_A_ENABLE.)

include ../common.mk
