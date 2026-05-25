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

# vortexpipe drives the Vortex graphics hardware units, so the SimX
# device build must enable them all: RASTER (Phase 4) + OM (Phase 5)
# + TEX (Phase 6). The macros below are what the canonical Vortex
# tree (VX_config.toml + gen_config.py) recognizes; the legacy
# `-DEXT_RASTER_ENABLE` shape (no VX_CFG_ prefix) was silently dropped
# and SimX was built without these extensions — every graphics test
# then crashed when its first Vortex op (vx_rast / vx_om / vx_tex)
# hit an unhandled-funct3 abort in the decoder.
CONFIGS += -DVX_CFG_EXT_RASTER_ENABLE \
           -DVX_CFG_EXT_OM_ENABLE \
           -DVX_CFG_EXT_TEX_ENABLE

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
# MESA_VORTEX_STRICT=1: vortexpipe refuses to silently fall back to llvmpipe
# (the CPU rasterizer it inherits from). With strict mode off, a missing
# kernel / runtime failure / NIR-translation gap was just a logw + CPU
# fallback, and the test would still PASS because llvmpipe correctly
# computed the result. Tests that EXPECT to run on Vortex set STRICT=1
# (the default below) and the harness fails on any fallback.
#
# A test that intentionally runs on llvmpipe (e.g. raytrace's lavapipe
# correctness oracle for future Vortex-SIMT RT phases) sets `STRICT := 0`
# in its own Makefile BEFORE including common.mk.
STRICT ?= 1
RUN_ENV := VK_ICD_FILENAMES=$(VK_ICD) \
           GALLIUM_DRIVER=vortexpipe \
           MESA_VORTEX_XLEN=$(XLEN) \
           MESA_VORTEX_STRICT=$(STRICT) \
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
#
# Two failure conditions the harness enforces (besides the test's own
# data-correctness check):
#   - any "MESA: error" line in stderr -> test fails. Catches every
#     mesa_loge in vortexpipe (toolchain failures, runtime API failures,
#     STRICT-mode fallback refusals).
#   - the device printed by the test must contain "vortex" -- guards
#     against the case where lavapipe selects a non-vortex ICD entirely
#     (Vortex's own ICD reports a Vortex device string; llvmpipe reports
#     "llvmpipe (LLVM …)").
# Without these checks the suite silently green-lighted CPU-only runs.
run run-simx: all $(VORTEX_RT_LIB)/libvortex.so
	$(RUNTIME_ARGS) $(MAKE) -C $(VORTEX_RT_SRC)/simx DESTDIR=$(VORTEX_RT_LIB)
	@out=`$(RUN_ENV) ./$(PROJECT) $(SPVS) $(OPTS) 2>&1`; status=$$?; \
	echo "$$out"; \
	if [ $$status -ne 0 ]; then \
	   echo "FAIL: $(PROJECT) exited $$status"; exit $$status; \
	fi; \
	if echo "$$out" | grep -q 'MESA: error'; then \
	   echo "FAIL: vortexpipe reported MESA: error (see above)"; exit 1; \
	fi; \
	if ! echo "$$out" | grep -qi 'device:.*vortex'; then \
	   echo "FAIL: $(PROJECT) did not report a Vortex device (likely fell back to llvmpipe)"; \
	   exit 1; \
	fi

clean:
	rm -rf $(PROJECT) $(SPVS) ramulator.stats.log trace *.dump *.log
