ROOT_DIR := $(realpath ../../..)
include $(ROOT_DIR)/config.mk

# cgo27_motivation exercises all 5 GEMM paths in one build:
#   in-core SIMT, in-core TCU, in-core TCU + DXA, DTCU, DTCU + DTCU_TMA.
# Needs the in-core TCU, the DTCU, and the DXA engine all enabled.
CONFIGS := $(if $(findstring -DVX_CFG_EXT_TCU_ENABLE,$(CONFIGS)),$(CONFIGS),$(CONFIGS) -DVX_CFG_EXT_TCU_ENABLE)
CONFIGS := $(if $(findstring -DVX_CFG_EXT_DTCU_ENABLE,$(CONFIGS)),$(CONFIGS),$(CONFIGS) -DVX_CFG_EXT_DTCU_ENABLE)
CONFIGS := $(if $(findstring -DVX_CFG_EXT_DXA_ENABLE,$(CONFIGS)),$(CONFIGS),$(CONFIGS) -DVX_CFG_EXT_DXA_ENABLE)
# Atomics: dtensor_check() reads the descriptor completion flag with an AMO, which
# invalidates the reader's stale L1 copy and resolves at the LLC. A plain load would
# cache the flag and spin on it forever.
CONFIGS := $(if $(findstring -DVX_CFG_EXT_A_ENABLE,$(CONFIGS)),$(CONFIGS),$(CONFIGS) -DVX_CFG_EXT_A_ENABLE)

# --- Experiment HW config (Hopper-proportional; ~1/4 H100 SM per core). ---
# See README.md for the scaling rationale. Override on the make/blackbox line
# to sweep (e.g. CONFIGS="-DVX_CFG_NUM_CORES=8").
# Workgroup MMA: lets a warpgroup read its operands straight out of Local Memory via a
# shared-memory descriptor instead of load_matrix_sync-ing them into registers. That is
# the third of the three things staging needs to pay (see README), and without it the DXA
# modes keep a full LSU load per fragment no matter how the copy arrives.
# Which epilogue this build contains. ONE per binary, chosen by the preprocessor, so an
# app the run does not use cannot relocate a mode's code in the icache. Override with
# `MOTI_APP=6 make ...`.
#
# DO NOT set CONFIGS= on the command line to pass one of these. A command-line variable
# cannot be appended to by the makefile, so every `CONFIGS +=` below -- the entire machine
# configuration, cores, warps, cache sizes, the lot -- is silently dropped, and the build
# comes out with the defaults. It does not fail; it hangs at run time. Each knob that needs
# overriding gets its own `?=` variable here instead.
MOTI_APP ?= 1
CONFIGS += -DMOTI_APP=$(MOTI_APP)
# Pipeline depth and poll backoff for the pipelined engine modes (14/15). See common.h.
MOTI_PIPE_TILES   ?= 4
MOTI_PIPE_BACKOFF ?= 32
MOTI_PIPE_C0_CONSUMES ?= 0
MOTI_PIPE_CONSUMER_WARPS ?= 16
MOTI_PIPE_VERIFY ?= 1
MOTI_PIPE_STRIPE_WARP ?= 0
CONFIGS += -DMOTI_PIPE_TILES=$(MOTI_PIPE_TILES) -DMOTI_PIPE_BACKOFF=$(MOTI_PIPE_BACKOFF)
CONFIGS += -DMOTI_PIPE_C0_CONSUMES=$(MOTI_PIPE_C0_CONSUMES)
CONFIGS += -DMOTI_PIPE_CONSUMER_WARPS=$(MOTI_PIPE_CONSUMER_WARPS) -DMOTI_PIPE_VERIFY=$(MOTI_PIPE_VERIFY) -DMOTI_PIPE_STRIPE_WARP=$(MOTI_PIPE_STRIPE_WARP)
CONFIGS += -DVX_CFG_TCU_WGMMA_ENABLE -DWGMMA_RS -DWGMMA_NRC=8

CONFIGS += -DVX_CFG_NUM_CLUSTERS=1
CONFIGS += -DVX_CFG_NUM_CORES=4
CONFIGS += -DVX_CFG_SOCKET_SIZE=1
CONFIGS := $(if $(findstring -DVX_CFG_NUM_WARPS=,$(CONFIGS)),$(CONFIGS),$(CONFIGS) -DVX_CFG_NUM_WARPS=16)
CONFIGS += -DVX_CFG_NUM_THREADS=32
CONFIGS += -DVX_CFG_ISSUE_WIDTH=4
CONFIGS += -DVX_CFG_NUM_BARRIERS=32          # pipelined double-buffer: 2 barriers/CTA
CONFIGS += -DVX_CFG_DCACHE_SIZE=32768        # 32 KB L1
CONFIGS += -DVX_CFG_LMEM_LOG_SIZE=16         # 64 KB shared memory (2^16)
CONFIGS += -DVX_CFG_L2_ENABLE                # enable L2
CONFIGS += -DVX_CFG_L2_CACHE_SIZE=1048576    # 1 MB L2

PROJECT := cgo27_motivation

SRC_DIR := $(VORTEX_HOME)/tests/regression/$(PROJECT)

SRCS := $(SRC_DIR)/main.cpp $(SW_COMMON_DIR)/rvfloats.cpp $(SW_COMMON_DIR)/softfloat_ext.cpp

# No combined device program any more. Every mode's kernel lives in its own
# kernel_m<N>.cpp (rules below); a kernel.cpp that also defined them would be a SECOND
# copy of the same nine kernels, free to drift from the one actually measured.
# VX_SRCS still has to name something for common.mk's kernel.elf rule, so it points at a
# stub that defines no entries.
VX_SRCS := $(SRC_DIR)/kernel_modes/kernel.cpp

# Layout: main.cpp + Makefile + sweep scripts at the top, everything else in a folder
# by side. common.h and epilogue.h stay at the top because they are the host/device ABI
# and both sides include them.
#   docs/          RFCs and the figures published as an Artifact
#   host/          x86 driver: mode registry, per-mode specs, run scaffolding, arg parsing
#   kernel_modes/  one GPU program per mode, plus the helpers more than one of them shares
CXXFLAGS += -I$(SRC_DIR) -I$(SRC_DIR)/host
CXXFLAGS += -I$(THIRD_PARTY_DIR)/softfloat/source/include

LDFLAGS += $(THIRD_PARTY_DIR)/softfloat/build/Linux-x86_64-GCC/softfloat.a

KERNEL_LIB := vortex2

include ../common.mk

# ---------------------------------------------------------------------------
# One device program per mode.
#
# All modes used to live in one kernel.vxbin. Only one runs at a time, but they all
# occupy address space, and address decides icache set -- adding modes 3/4 moved mode 2
# from 15,468 to 24,106 cycles with a byte-identical kernel body (k_select.h has the
# counters). Building each mode alone makes its code start at the same low address
# whatever else exists in the tree, so a measurement is a property of the mode.
#
# common.mk's own kernel.vxbin (from kernel.cpp, all modes) is still built and is the
# cross-check: a per-mode number should differ from the combined one only by layout.
# ---------------------------------------------------------------------------
# A per-mode program includes shared helper headers (wmma_common.h) and by construction
# uses only some of them, so -Wunused-function fires on helpers another mode needs. That
# is the intended shape here, not a defect -- suppress it for these targets only, leaving
# the warning active for the all-in-one kernel.cpp and the host.
MOTI_VX_CFLAGS := $(VX_CFLAGS) -I$(SRC_DIR) -I$(SRC_DIR)/kernel_modes -Wno-unused-function

MOTI_MODES  := 0 1 2 3 4 5 7 8 14 15
MOTI_VXBINS := $(foreach m,$(MOTI_MODES),kernel_m$(m).vxbin)

all: $(MOTI_VXBINS)

# The runner loads kernel_m<mode>.vxbin, so every run target needs them present. Without
# this, deleting a per-mode binary makes the harness fail at vx_module_load_file instead
# of rebuilding it -- common.mk's own run-simx only knows about kernel.vxbin.
run-simx run-rtlsim run-opae run-xrt: $(MOTI_VXBINS)

kernel_m%.vxbin: kernel_m%.elf
	OBJCOPY=$(VX_CP) $(VORTEX_HOME)/sw/kernel/scripts/vxbin.py $< $@

# Mirrors common.mk's kernel.elf recipe, including its two-pass startup sizing: the
# first link only exists so KERNEL_STARTUP can measure it and rebuild vx_start with the
# right constants. Per-mode object names so the modes never collide in this directory.
kernel_m%.elf: $(SRC_DIR)/kernel_modes/kernel_m%.cpp $(VORTEX_KN_PATH)/lib$(KERNEL_LIB).a $(CONFIG_STAMP)
	$(VX_CXX) $(MOTI_VX_CFLAGS) -c $< -o kernel_m$*.o
	$(VX_CXX) $(MOTI_VX_CFLAGS) -DNEED_GP -DNEED_TLS -DNEED_INITFINI $(VX_KMU_FLAG) -c $(VX_STARTUP_SRC) -o vx_start_m$*.o
	$(VX_CXX) $(MOTI_VX_CFLAGS) vx_start_m$*.o kernel_m$*.o $(VX_LDFLAGS) -o $@.probe
	$(VX_CXX) $(MOTI_VX_CFLAGS) $$($(KERNEL_STARTUP) $(VX_DP) $@.probe) $(VX_KMU_FLAG) -c $(VX_STARTUP_SRC) -o vx_start_m$*.o
	$(VX_CXX) $(MOTI_VX_CFLAGS) vx_start_m$*.o kernel_m$*.o $(VX_LDFLAGS) -o $@
	rm -f $@.probe

# `make sizes` -- what each mode's program actually costs against the 16 KB icache.
sizes: $(foreach m,$(MOTI_MODES),kernel_m$(m).elf)
	@printf '%-18s %8s  %s\n' program .text 'of 16 KB icache' ; \
	 for f in $^ ; do \
	   sz=$$($(VX_DP) -h $$f | awk '/\.text/{print strtonum("0x"$$3)}') ; \
	   printf '%-18s %8d  %5.1f%%\n' $$f $$sz $$(echo "$$sz 16384" | awk '{print 100*$$1/$$2}') ; \
	 done

clean-modes:
	rm -f kernel_m*.o kernel_m*.elf kernel_m*.vxbin kernel_m*.probe vx_start_m*.o

.PHONY: sizes clean-modes
