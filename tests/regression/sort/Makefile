PROJECT = sort

SRCS = main.cpp

VX_SRCS = kernel.c

OPTS ?= -n16

include ../common.mk

VX_CC  = $(LLVM_PREFIX)/bin/clang $(LLVM_CFLAGS)
VX_CXX = $(LLVM_PREFIX)/bin/clang++ -std=c++17 $(LLVM_CFLAGS)
VX_DP  = $(LLVM_PREFIX)/bin/llvm-objdump
VX_CP  = $(LLVM_PREFIX)/bin/llvm-objcopy