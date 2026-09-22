ROOT_DIR := $(realpath ../../../..)
include $(ROOT_DIR)/config.mk
# exports VERILATOR/VERILATOR_PATH, which gen_sources.sh resolves from the
# environment rather than PATH
include $(VORTEX_HOME)/hw/syn/common.mk

ifneq ($(findstring Makefile, $(MAKEFILE_LIST)), Makefile)
help:
	$(ECHO) "Makefile Usage:"
	$(ECHO) "  make all TARGET=<hw|sim>"
	$(ECHO) "      Package the AFU as IP-XACT and link a V80 device binary."
	$(ECHO) ""
	$(ECHO) "  make clean"
	$(ECHO) "      Remove generated files."
	$(ECHO) ""
endif

TARGET ?= hw

PREFIX ?= build$(XLEN)

# V80 device. Override for a different speed grade or package.
DEVICE ?= xcv80-lsva4737-2MHP-e-S

SRC_DIR := $(VORTEX_HOME)/hw/syn/xilinx/aved
AFU_DIR := $(VORTEX_HOME)/hw/rtl/afu/aved

SLASHKIT ?= slashkit

BUILD_DIR := $(PREFIX)_aved_$(TARGET)
BIN_DIR := $(BUILD_DIR)/bin

IP_DIR := $(BUILD_DIR)/vortex_afu_ip
VBIN_CONTAINER := $(BIN_DIR)/vortex_afu.vbin

# RTL kernels are only supported by the hardware and simulation platforms.
# The emulation platform is a behavioural C-model that does not accept them.
ifeq ($(TARGET), emu)
$(error TARGET=emu is unavailable: the V80 emulation platform is a C-model \
        that does not accept RTL kernels. Use TARGET=avedsim for fast \
        functional iteration, or --driver=simx for the C-model)
endif

ifneq ($(TARGET),$(findstring $(TARGET), hw sim))
$(error TARGET must be hw or sim (got '$(TARGET)'))
endif

ifeq ($(TARGET), sim)
	override CONFIGS += -DSIMULATION
endif

.DEFAULT_GOAL := all
include $(VORTEX_HOME)/hw/syn/xilinx/common.mk

# Card settings (memory geometry, connectivity tags, kernel clock), resolved
# from the device part. A tag the card refuses -- on the V80, HOST, whose reads
# never complete on the compute shell -- stops the build at parse time rather
# than after a full synthesis run.
PLATFORM_GEN_ARGS := --flow=aved --device=$(DEVICE)
include $(VORTEX_HOME)/hw/syn/xilinx/platform.mk

# MEM covers the 32-bit device space from the HBM base, i.e. HBM0..HBM7; a CP
# staging aperture there aliases device memory and large uploads overwrite the
# heap. See hw/syn/xilinx/platforms/v80.toml.
ifneq ($(filter HBM0 HBM1 HBM2 HBM3 HBM4 HBM5 HBM6 HBM7,$(HOST_TAG)),)
ifneq ($(filter MEM HBM0,$(MEM_TAG)),)
$(error HOST_TAG=$(HOST_TAG) lies inside the device-memory window of MEM_TAG=$(MEM_TAG) (HBM0..HBM7); CP staging would alias the Vortex heap. Use HBM8 or above)
endif
endif

# Debugging
ifdef DEBUG
	CFLAGS += -DVX_DBG_DEBUG_LEVEL=$(DEBUG) $(DBG_TRACE_FLAGS)
else
	CFLAGS += -DNDEBUG
endif

CFLAGS += $(COMPILE_CFLAGS)

.PHONY: all clean package-ip gen-bin FORCE

all: check-axil-blocks $(VBIN_CONTAINER)

# Regenerated on every run and rewritten only when its content changes, so a
# command-line tag or clock override reaches the linker without invalidating an
# unchanged build.
$(BUILD_DIR)/config.cfg: FORCE
	$(PLATFORM_GEN) --emit=cfg --mem-tag='$(MEM_TAG)' --host-tag='$(HOST_TAG)' --kernel-mhz=$(KERNEL_FREQ) -o $@

FORCE:

package-ip: $(IP_DIR)/component.xml
$(IP_DIR)/component.xml: $(BUILD_DIR)/sources.txt
	cd $(BUILD_DIR); TOOL_DIR=$(SCRIPT_DIR) $(VIVADO) -mode batch -source $(SRC_DIR)/package_ip.tcl -tclargs vortex_afu sources.txt . $(DEVICE)

gen-bin: $(VBIN_CONTAINER)
# TOOL_DIR is consumed by pre_opt_hook.tcl to locate hw/scripts. The pre-synth
# hook is the linker's only opportunity to set per-step properties on impl_1,
# and it is what attaches the pre-opt hook that resolves the async-BRAM
# placeholders.
$(VBIN_CONTAINER): $(IP_DIR)/component.xml $(BUILD_DIR)/config.cfg $(SCOPE_JSON)
	mkdir -p $(BIN_DIR)
	TOOL_DIR=$(SCRIPT_DIR) $(SLASHKIT) link -c $(BUILD_DIR)/config.cfg -p $(TARGET) -o $(VBIN_CONTAINER) \
	  -k $(IP_DIR)/component.xml --vivado $(VIVADO) \
	  --pre-synth-tcls $(SRC_DIR)/pre_synth_hook.tcl

clean:
	$(RMDIR) $(BUILD_DIR)
