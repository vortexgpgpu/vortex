ROOT_DIR := $(realpath ../../..)
include $(ROOT_DIR)/config.mk

# llama_dtcu: llama2.c inference where every GEMM runs on one of the cgo27_motivation
# execution modes (0-8, 14, 15) through the microbenchmark's OWN per-mode device programs
# (kernel_m<N>.vxbin, built here from cgo27_motivation/kernel_modes without modification).
#
# The machine configuration below is the microbenchmark's, line for line, so a GEMM
# measured here and the same GEMM measured by the microbenchmark run on the same simulated
# machine. Never pass CONFIGS= on the make command line (see cgo27_motivation/Makefile).
CONFIGS := $(if $(findstring -DVX_CFG_EXT_TCU_ENABLE,$(CONFIGS)),$(CONFIGS),$(CONFIGS) -DVX_CFG_EXT_TCU_ENABLE)
CONFIGS := $(if $(findstring -DVX_CFG_EXT_DTCU_ENABLE,$(CONFIGS)),$(CONFIGS),$(CONFIGS) -DVX_CFG_EXT_DTCU_ENABLE)
CONFIGS := $(if $(findstring -DVX_CFG_EXT_DXA_ENABLE,$(CONFIGS)),$(CONFIGS),$(CONFIGS) -DVX_CFG_EXT_DXA_ENABLE)
CONFIGS := $(if $(findstring -DVX_CFG_EXT_A_ENABLE,$(CONFIGS)),$(CONFIGS),$(CONFIGS) -DVX_CFG_EXT_A_ENABLE)

# Only the baseline epilogue is ever compiled into the mode programs: llama's residual is
# the C preload (D = x + A*B), and every other non-GEMM op is a separate pass.
MOTI_APP := 1
CONFIGS += -DMOTI_APP=$(MOTI_APP)
MOTI_PIPE_TILES   ?= 4
MOTI_PIPE_BACKOFF ?= 32
MOTI_PIPE_CONSUMER_WARPS ?= 15
MOTI_PIPE_STRIPE_WARP ?= 0
MOTI_L2_ARB_ENGINE_BYPASS_LIMIT ?= 0
CONFIGS += -DMOTI_PIPE_TILES=$(MOTI_PIPE_TILES) -DMOTI_PIPE_BACKOFF=$(MOTI_PIPE_BACKOFF)
CONFIGS += -DMOTI_PIPE_CONSUMER_WARPS=$(MOTI_PIPE_CONSUMER_WARPS) -DMOTI_PIPE_STRIPE_WARP=$(MOTI_PIPE_STRIPE_WARP)
CONFIGS += -DVX_CFG_L2_ARB_ENGINE_BYPASS_LIMIT=$(MOTI_L2_ARB_ENGINE_BYPASS_LIMIT)
CONFIGS += -DVX_CFG_TCU_WGMMA_ENABLE -DWGMMA_RS -DWGMMA_NRC=8

CONFIGS += -DVX_CFG_NUM_CLUSTERS=1
CONFIGS += -DVX_CFG_NUM_CORES=4
CONFIGS += -DVX_CFG_SOCKET_SIZE=1
CONFIGS := $(if $(findstring -DVX_CFG_NUM_WARPS=,$(CONFIGS)),$(CONFIGS),$(CONFIGS) -DVX_CFG_NUM_WARPS=16)
CONFIGS += -DVX_CFG_NUM_THREADS=32
CONFIGS += -DVX_CFG_ISSUE_WIDTH=4
CONFIGS += -DVX_CFG_NUM_BARRIERS=32
CONFIGS += -DVX_CFG_DCACHE_SIZE=32768
CONFIGS += -DVX_CFG_LMEM_LOG_SIZE=16
CONFIGS += -DVX_CFG_L2_ENABLE
CONFIGS += -DVX_CFG_L2_CACHE_SIZE=1048576
LLAMA_EXTRA_CONFIGS ?=
CONFIGS += $(LLAMA_EXTRA_CONFIGS)

PROJECT := llama_dtcu

SRC_DIR  := $(VORTEX_HOME)/tests/regression/$(PROJECT)
MOTI_DIR := $(VORTEX_HOME)/tests/regression/cgo27_motivation

SRCS := $(SRC_DIR)/main.cpp $(SRC_DIR)/llama_dev.cpp $(SRC_DIR)/llama_forward.cpp $(SRC_DIR)/llama_decode.cpp \
        $(SW_COMMON_DIR)/rvfloats.cpp $(SW_COMMON_DIR)/softfloat_ext.cpp

# common.mk's kernel.elf rule needs VX_SRCS to name something; the microbenchmark's stub
# defines no entries. The real device programs are the per-mode vxbins below.
VX_SRCS := $(MOTI_DIR)/kernel_modes/kernel.cpp

CXXFLAGS += -I$(SRC_DIR) -I$(MOTI_DIR) -I$(MOTI_DIR)/host
CXXFLAGS += -I$(THIRD_PARTY_DIR)/softfloat/source/include
# The microbenchmark's host headers define file-scope statics its main.cpp uses and this
# host does not (g_app, mode_state, ...). Unused-warnings for those are not defects here.
CXXFLAGS += -Wno-unused-function -Wno-unused-variable

LDFLAGS += $(THIRD_PARTY_DIR)/softfloat/build/Linux-x86_64-GCC/softfloat.a

KERNEL_LIB := vortex2

# Default run: the GEMM probe (one harness-style GEMM through one mode, verified).
OPTS ?= --probe -m 1 -M 128 -N 288 -K 288

include ../common.mk

# ---------------------------------------------------------------------------
# One device program per mode, built from the microbenchmark's sources, byte for byte the
# same programs the microbenchmark measures with (same sources, same flags, MOTI_APP=1).
# ---------------------------------------------------------------------------
MOTI_VX_CFLAGS := $(VX_CFLAGS) -I$(MOTI_DIR) -I$(MOTI_DIR)/kernel_modes -Wno-unused-function

MOTI_MODES  := 0 1 2 3 4 5 6 7 8 14 15
MOTI_VXBINS := $(foreach m,$(MOTI_MODES),kernel_m$(m).vxbin)

# The llama forward's non-GEMM passes, in their own program (see kernel_passes.cpp).
PASS_VXBIN := llama_passes.vxbin

all: $(MOTI_VXBINS) $(PASS_VXBIN)

run-simx run-rtlsim run-opae run-xrt: $(MOTI_VXBINS) $(PASS_VXBIN)

llama_passes.vxbin: llama_passes.elf
	OBJCOPY=$(VX_CP) $(VORTEX_HOME)/sw/kernel/scripts/vxbin.py $< $@

llama_passes.elf: $(SRC_DIR)/kernel_passes.cpp $(SRC_DIR)/llama_passes.h $(VORTEX_KN_PATH)/lib$(KERNEL_LIB).a $(CONFIG_STAMP)
	$(VX_CXX) $(MOTI_VX_CFLAGS) -I$(SRC_DIR) -c $< -o llama_passes.o
	$(VX_CXX) $(MOTI_VX_CFLAGS) -DNEED_GP -DNEED_TLS -DNEED_INITFINI $(VX_KMU_FLAG) -c $(VX_STARTUP_SRC) -o vx_start_passes.o
	$(VX_CXX) $(MOTI_VX_CFLAGS) vx_start_passes.o llama_passes.o $(VX_LDFLAGS) -o $@.probe
	$(VX_CXX) $(MOTI_VX_CFLAGS) $$($(KERNEL_STARTUP) $(VX_DP) $@.probe) $(VX_KMU_FLAG) -c $(VX_STARTUP_SRC) -o vx_start_passes.o
	$(VX_CXX) $(MOTI_VX_CFLAGS) vx_start_passes.o llama_passes.o $(VX_LDFLAGS) -o $@
	rm -f $@.probe

kernel_m%.vxbin: kernel_m%.elf
	OBJCOPY=$(VX_CP) $(VORTEX_HOME)/sw/kernel/scripts/vxbin.py $< $@

kernel_m%.elf: $(MOTI_DIR)/kernel_modes/kernel_m%.cpp $(VORTEX_KN_PATH)/lib$(KERNEL_LIB).a $(CONFIG_STAMP)
	$(VX_CXX) $(MOTI_VX_CFLAGS) -c $< -o kernel_m$*.o
	$(VX_CXX) $(MOTI_VX_CFLAGS) -DNEED_GP -DNEED_TLS -DNEED_INITFINI $(VX_KMU_FLAG) -c $(VX_STARTUP_SRC) -o vx_start_m$*.o
	$(VX_CXX) $(MOTI_VX_CFLAGS) vx_start_m$*.o kernel_m$*.o $(VX_LDFLAGS) -o $@.probe
	$(VX_CXX) $(MOTI_VX_CFLAGS) $$($(KERNEL_STARTUP) $(VX_DP) $@.probe) $(VX_KMU_FLAG) -c $(VX_STARTUP_SRC) -o vx_start_m$*.o
	$(VX_CXX) $(MOTI_VX_CFLAGS) vx_start_m$*.o kernel_m$*.o $(VX_LDFLAGS) -o $@
	rm -f $@.probe

clean-modes:
	rm -f kernel_m*.o kernel_m*.elf kernel_m*.vxbin kernel_m*.probe vx_start_m*.o llama_passes.* vx_start_passes.o

.PHONY: clean-modes
