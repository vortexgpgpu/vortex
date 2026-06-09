# Copyright © 2026  Vortex GPGPU
# SPDX-License-Identifier: MIT
#
# Shared build rules for tests/vulkan/*.
#
# Unlike tests/opencl Vulkan test is an # ordinary host program:
# it links the Vulkan loader and drives the
# Vortex GPU through Mesa's lavapipe ICD and the vortexpipe Gallium
# driver. The RISC-V kernel is generated inside vortexpipe at run
# time (NIR -> LLVM -> .vxbin), so no RISC-V toolchain build is
# needed here -- only glslc, to turn the GLSL compute shaders into
# the SPIR-V the test feeds to Vulkan.
#
# A per-test Makefile sets PROJECT, SRC_DIR, SRCS and SHADERS, then
# includes this file -- see tests/vulkan/compute/Makefile.

# Mesa-with-vortexpipe install + supporting toolchain libraries.
MESA_PATH ?= $(TOOLDIR)/mesa-vortex
ZSTD_PATH ?= $(TOOLDIR)/zstd

# Vortex runtime, built out-of-tree under the configure build dir.
# Mesa's libgallium links libvortex.so; at run time it dlopens
# libvortex-<NAME>.so via an $ORIGIN rpath to select the backend.
VORTEX_RT_SRC ?= $(ROOT_DIR)/sw/runtime
VORTEX_RT_LIB ?= $(VORTEX_RT_SRC)

# FPGA/HW backend selectors — same names + defaults as tests/regression.
#   opae : TARGET picks the libopae shape (real FPGA vs ASE/opaesim emulators)
#   xrt  : TARGET picks among Xilinx hw / hw_emu / sw_emu;
#          FPGA_BIN_DIR points at the built xclbin tree for hw/hw_emu.
TARGET ?= opaesim
XRT_SYN_DIR ?= $(VORTEX_HOME)/hw/syn/xilinx/xrt
XRT_DEVICE_INDEX ?= 0

ifeq ($(TARGET), fpga)
    OPAE_DRV_PATHS ?= libopae-c.so
else ifeq ($(TARGET), asesim)
    OPAE_DRV_PATHS ?= libopae-c-ase.so
else ifeq ($(TARGET), opaesim)
    OPAE_DRV_PATHS ?= libopae-c-sim.so
endif

# Enable all graphics hardware units required by vortexpipe.
# Use the VX_CFG_EXT_* macro names recognized by VX_config.toml + gen_config.py.
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
VK_ICD  := $(MESA_PATH)/share/vulkan/icd.d/lvp_icd.x86_64.json
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
# Common env every run-* recipe sets. The per-backend recipes below each
# prepend backend-specific entries to LD_LIBRARY_PATH (e.g. $XILINX_XRT/lib
# for run-xrt) and add extra env vars (SCOPE_JSON_PATH, OPAE_DRV_PATHS,
# XCL_EMULATION_MODE, …) inline, matching tests/opencl/common.mk's shape.
RUN_ENV := VK_ICD_FILENAMES=$(VK_ICD) \
           GALLIUM_DRIVER=vortexpipe \
           MESA_VORTEX_XLEN=$(XLEN) \
           MESA_VORTEX_STRICT=$(STRICT) \
           VORTEX_HOME=$(VORTEX_HOME) \
           VORTEX_TOOLDIR=$(TOOLDIR) \
           VORTEX_BUILD=$(ROOT_DIR)
RUN_LD_PATH := $(MESA_PATH)/lib:$(LLVM_PATH)/lib:$(ZSTD_PATH)/lib:$(VORTEX_RT_LIB)

RUNTIME_ARGS = CONFIGS="$(CONFIGS)" $(if $(DEBUG),DEBUG=$(DEBUG)) $(if $(PERF),PERF=$(PERF)) $(if $(SCOPE),SCOPE=$(SCOPE))

# Post-run sanity gates, applied to every run-* recipe via $(call check_run,$out,$status).
# Catches the silent-green-light failure modes:
#   - any "MESA: error" line in stderr → fail (every mesa_loge in vortexpipe:
#     toolchain failures, runtime API failures, STRICT-mode fallback refusals).
#   - the device printed by the test must contain "vortex" → guards against
#     lavapipe silently selecting a non-vortex ICD (llvmpipe reports
#     "llvmpipe (LLVM …)").
define check_run
	status=$$?; \
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
endef

.PHONY: all run run-simx run-rtlsim run-opae run-xrt clean

all: $(PROJECT) $(SPVS)

$(PROJECT): $(SRCS)
	$(CC) $(CFLAGS) $(SRCS) $(LDLIBS) -o $@

# glslc infers the shader stage from the source extension.
%.spv: $(SRC_DIR)/%
	$(GLSLC) $< -o $@

# libvortex.so: the vortex2.h API layer Mesa's libgallium links.
$(VORTEX_RT_LIB)/libvortex.so:
	$(RUNTIME_ARGS) $(MAKE) -C $(VORTEX_RT_SRC)/stub DESTDIR=$(VORTEX_RT_LIB)

# `run` defaults to the SimX backend; explicit recipes select simx / rtlsim
# / opae / xrt. vortexpipe is backend-agnostic — same .vxbin, the stub
# libvortex.so dlopens libvortex-<VORTEX_DRIVER>.so at runtime.

run run-simx: all $(VORTEX_RT_LIB)/libvortex.so
	$(RUNTIME_ARGS) $(MAKE) -C $(VORTEX_RT_SRC)/simx DESTDIR=$(VORTEX_RT_LIB)
	@out=`$(RUN_ENV) LD_LIBRARY_PATH=$(RUN_LD_PATH) VORTEX_DRIVER=simx ./$(PROJECT) $(SPVS) $(OPTS) 2>&1`; \
	$(check_run)

run-rtlsim: all $(VORTEX_RT_LIB)/libvortex.so
	$(RUNTIME_ARGS) $(MAKE) -C $(VORTEX_RT_SRC)/rtlsim DESTDIR=$(VORTEX_RT_LIB)
	@out=`$(RUN_ENV) LD_LIBRARY_PATH=$(RUN_LD_PATH) VORTEX_DRIVER=rtlsim ./$(PROJECT) $(SPVS) $(OPTS) 2>&1`; \
	$(check_run)

run-opae: all $(VORTEX_RT_LIB)/libvortex.so
	$(RUNTIME_ARGS) $(MAKE) -C $(VORTEX_RT_SRC)/opae DESTDIR=$(VORTEX_RT_LIB)
	@out=`SCOPE_JSON_PATH=$(VORTEX_RT_LIB)/scope.json OPAE_DRV_PATHS=$(OPAE_DRV_PATHS) $(RUN_ENV) LD_LIBRARY_PATH=$(RUN_LD_PATH) VORTEX_DRIVER=opae ./$(PROJECT) $(SPVS) $(OPTS) 2>&1`; \
	$(check_run)

run-xrt: all $(VORTEX_RT_LIB)/libvortex.so
	$(RUNTIME_ARGS) $(MAKE) -C $(VORTEX_RT_SRC)/xrt DESTDIR=$(VORTEX_RT_LIB)
ifeq ($(TARGET), hw)
	@out=`SCOPE_JSON_PATH=$(FPGA_BIN_DIR)/scope.json XRT_INI_PATH=$(VORTEX_RT_SRC)/xrt/xrt.ini EMCONFIG_PATH=$(FPGA_BIN_DIR) XRT_DEVICE_INDEX=$(XRT_DEVICE_INDEX) XRT_XCLBIN_PATH=$(FPGA_BIN_DIR)/vortex_afu.xclbin $(RUN_ENV) LD_LIBRARY_PATH=$(XILINX_XRT)/lib:$(RUN_LD_PATH) VORTEX_DRIVER=xrt ./$(PROJECT) $(SPVS) $(OPTS) 2>&1`; \
	$(check_run)
else ifeq ($(TARGET), hw_emu)
	@out=`SCOPE_JSON_PATH=$(FPGA_BIN_DIR)/scope.json XCL_EMULATION_MODE=$(TARGET) XRT_INI_PATH=$(VORTEX_RT_SRC)/xrt/xrt.ini EMCONFIG_PATH=$(FPGA_BIN_DIR) XRT_DEVICE_INDEX=$(XRT_DEVICE_INDEX) XRT_XCLBIN_PATH=$(FPGA_BIN_DIR)/vortex_afu.xclbin $(RUN_ENV) LD_LIBRARY_PATH=$(XILINX_XRT)/lib:$(RUN_LD_PATH) VORTEX_DRIVER=xrt ./$(PROJECT) $(SPVS) $(OPTS) 2>&1`; \
	$(check_run)
else
	@out=`SCOPE_JSON_PATH=$(VORTEX_RT_LIB)/scope.json $(RUN_ENV) LD_LIBRARY_PATH=$(XILINX_XRT)/lib:$(RUN_LD_PATH) VORTEX_DRIVER=xrt ./$(PROJECT) $(SPVS) $(OPTS) 2>&1`; \
	$(check_run)
endif

clean:
	rm -rf $(PROJECT) $(SPVS) ramulator.stats.log trace *.dump *.log
