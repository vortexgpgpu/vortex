# Copyright © 2026  Vortex GPGPU
# SPDX-License-Identifier: MIT
#
# Shared build rules for tests/vulkan/*.
#
# Unlike tests/opencl and tests/regression, a Vulkan test is an
# ordinary host program: it links the Vulkan loader and drives the
# Vortex GPU through Mesa's lavapipe ICD and the vortexpipe Gallium
# driver. The RISC-V kernel is generated inside vortexpipe at run
# time (NIR -> LLVM -> .vxbin), so no RISC-V toolchain build is
# needed here -- only glslc, to turn the GLSL compute shaders into
# the SPIR-V the test feeds to Vulkan.
#
# A per-test Makefile sets PROJECT, SRC_DIR, SRCS and SHADERS, then
# includes this file -- see tests/vulkan/compute/Makefile.

# Mesa-with-vortexpipe install + supporting toolchain libraries.
MESA_VORTEX ?= $(TOOLDIR)/mesa-vortex
ZSTD        ?= $(TOOLDIR)/zstd

# Vortex runtime, built out-of-tree under the configure build dir.
# Mesa's libgallium links libvortex.so (the backend-agnostic
# vortex2.h API layer); at run time it dlopens libvortex-simx.so,
# which finds libsimx.so via an $ORIGIN rpath.
VORTEX_RT_SRC ?= $(ROOT_DIR)/sw/runtime
VORTEX_RT_LIB ?= $(VORTEX_RT_SRC)

GLSLC ?= glslc
CC    ?= cc

CFLAGS += -O2 -Wall
LDLIBS += -lvulkan

# One SPIR-V module per shader source, stage kept in the name so a
# test's .vert/.frag/.comp never collide (foo.vert -> foo.vert.spv).
SPVS := $(addsuffix .spv, $(notdir $(SHADERS)))

# Run the test under the lavapipe ICD with the vortexpipe driver
# selected; LD_LIBRARY_PATH covers Mesa, host LLVM, zstd and the
# Vortex runtime. VORTEX_HOME / VORTEX_TOOLDIR tell vortexpipe's
# .vxbin compiler where the Vortex tree and toolchain are -- needed
# because a prebuilt Mesa's baked-in paths point at the build host.
VK_ICD  := $(MESA_VORTEX)/share/vulkan/icd.d/lvp_icd.x86_64.json
RUN_ENV := VK_ICD_FILENAMES=$(VK_ICD) \
           GALLIUM_DRIVER=vortexpipe \
           VORTEX_HOME=$(VORTEX_HOME) \
           VORTEX_TOOLDIR=$(TOOLDIR) \
           VORTEX_BUILD=$(ROOT_DIR) \
           LD_LIBRARY_PATH=$(MESA_VORTEX)/lib:$(LLVM_VORTEX)/lib:$(ZSTD)/lib:$(VORTEX_RT_LIB)

RUNTIME_ARGS = CONFIGS="$(CONFIGS)" $(if $(DEBUG),DEBUG=$(DEBUG))

.PHONY: all run run-simx clean

all: $(PROJECT) $(SPVS)

$(PROJECT): $(SRCS)
	$(CC) $(CFLAGS) $(SRCS) $(LDLIBS) -o $@

# glslc infers the shader stage from the source extension.
%.spv: $(SRC_DIR)/%
	$(GLSLC) $< -o $@

# libvortex.so: the vortex2.h API layer Mesa's libgallium links.
$(VORTEX_RT_LIB)/libvortex.so:
	$(RUNTIME_ARGS) $(MAKE) -C $(VORTEX_RT_SRC)/stub DESTDIR=$(VORTEX_RT_LIB)

# vortexpipe currently always targets the Vortex SimX backend, so
# `run` and `run-simx` share a recipe; run-simx matches the naming
# the opencl/regression suites use.
run run-simx: all $(VORTEX_RT_LIB)/libvortex.so
	$(RUNTIME_ARGS) $(MAKE) -C $(VORTEX_RT_SRC)/simx DESTDIR=$(VORTEX_RT_LIB)
	$(RUN_ENV) ./$(PROJECT) $(SPVS) $(OPTS)

clean:
	rm -rf $(PROJECT) $(SPVS) ramulator.stats.log trace *.dump *.log
