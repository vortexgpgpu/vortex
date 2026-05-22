# Copyright © 2026  Vortex GPGPU
# SPDX-License-Identifier: MIT
#
# Minimal Vulkan compute smoke test for the vortexpipe driver.
# Build:  make
# Run:    make run     (selects the vortexpipe Gallium driver)

TOOLDIR     ?= $(HOME)/tools
VORTEX_HOME ?= $(abspath ../../..)
MESA_VORTEX ?= $(TOOLDIR)/mesa-vortex
LLVM_VORTEX ?= $(TOOLDIR)/llvm-vortex
ZSTD        ?= $(TOOLDIR)/zstd

GLSLC ?= glslc
CC    ?= cc
CFLAGS  += -O2 -Wall
LDLIBS  += -lvulkan

TARGET := vk_compute
SHADER := add1.spv

# The Vortex SimX runtime (libvortex.so + libvortex-simx.so + libsimx.so)
RT_DIR := $(VORTEX_HOME)/build/sw/runtime

VK_ICD  := $(MESA_VORTEX)/share/vulkan/icd.d/lvp_icd.x86_64.json
RUN_ENV := VK_ICD_FILENAMES=$(VK_ICD) \
           GALLIUM_DRIVER=vortexpipe \
           LD_LIBRARY_PATH=$(MESA_VORTEX)/lib:$(LLVM_VORTEX)/lib:$(ZSTD)/lib:$(RT_DIR)

.PHONY: all run clean

all: $(TARGET) $(SHADER)

$(TARGET): main.c
	$(CC) $(CFLAGS) $< $(LDLIBS) -o $@

$(SHADER): add1.comp
	$(GLSLC) $< -o $@

run: all
	$(RUN_ENV) ./$(TARGET) $(SHADER)

clean:
	rm -rf $(TARGET) $(SHADER) ramulator.stats.log trace *.dump
