ROOT_DIR := $(realpath ../../..)
include $(ROOT_DIR)/config.mk

CLOCK_FREQ	?= 800
DELAY_UNC 	?= 0.02
DELAY_IO  	?= 0.05
SAIF_FILE 	?=
SAIF_INST 	?=
LIB_TYPE 	?= DEFAULT

# Synthesis optimization level (standardized across hw/syn):
#   0 -- compile -map_effort low    (fastest, lowest QoR)
#   1 -- compile -map_effort medium
#   2 -- compile_ultra              (no retiming)
#   3 -- compile_ultra -retime      (default, highest QoR)
# project.tcl reads this through the OPT_LEVEL env var.
OPT_LEVEL ?= 3

SRC_DIR := $(VORTEX_HOME)/hw/syn/synopsys
LIB_DIR := $(VORTEX_HOME)/hw/syn/libs

ASAP7_ROOT      := /mnt/nas0/eda.libs/asap7
ASAP7_LIB_ROOT  := $(ASAP7_ROOT)/asap7sc7p5t_28/LIB/NLDM
ASAP7_SRAM_LIB  := $(ASAP7_ROOT)/ASAP7_SRAM_0p0/generated/LIB/srambank_128x4x32_6t122.lib
ASAP7_SRAM_DB   := $(ASAP7_ROOT)/ASAP7_SRAM_0p0/generated/LIB/srambank_128x4x32_6t122.db
ASAP7_NAME      := "ASAP7 (7nm)"

SAED14_ROOT     := /mnt/nas0/eda.libs/saed14/EDK_03_2025
SAED14_LIB_ROOT := $(SAED14_ROOT)/SAED14nm_EDK_STD_SLVT/liberty/nldm
SAED14_SRAM_LIB := $(SAED14_ROOT)/SAED14nm_EDK_SRAM/liberty/nldm/saed14sram_tt0p8v25c.lib
SAED14_SRAM_DB  := $(SAED14_ROOT)/SAED14nm_EDK_SRAM/liberty/nldm/saed14sram_tt0p8v25c.db
SAED14_NAME     := "SAED14 (14nm SLVT)"

DEFAULT_ROOT     :=
DEFAULT_LIB_ROOT :=
DEFAULT_SRAM_LIB :=
DEFAULT_SRAM_DB  :=
DEFAULT_NAME     := "NanGate 15nm OCL"

LIB_ROOT := $($(LIB_TYPE)_LIB_ROOT)
SRAM_LIB := $($(LIB_TYPE)_SRAM_LIB)
SRAM_DB  := $($(LIB_TYPE)_SRAM_DB)
LIB_NAME := $($(LIB_TYPE)_NAME)

ifeq ($(LIB_NAME),)
    $(error Invalid LIB_TYPE=$(LIB_TYPE).)
endif

$(info [Make] Using library $(LIB_NAME))

DEFAULT_LIB := $(LIB_DIR)/NanGate_15nm_OCL.db
SYN_LIB_ENV := $(if $(LIB_TGT),LIB_TGT=$(LIB_TGT),$(if $(LIB_ROOT),LIB_ROOT=$(LIB_ROOT),LIB_TGT=$(DEFAULT_LIB)))

TOP_LEVEL_ENTITY ?= Vortex
PREFIX ?= build

SCRIPT_DIR := $(VORTEX_HOME)/hw/scripts
RTL_DIR := $(VORTEX_HOME)/hw/rtl

WALL_IGNORE := "*/stream_omega_net.sv,*/fpnew_pkg.sv"

CP = cp -rf
RMDIR = rm -rf
ECHO = @echo

BUILD_DIR := $(PREFIX)_$(TOP_LEVEL_ENTITY)
BIN_DIR := $(BUILD_DIR)/bin

# control RTL debug tracing states
DBG_TRACE_FLAGS += -DDBG_TRACE_PIPELINE
DBG_TRACE_FLAGS += -DDBG_TRACE_MEM
DBG_TRACE_FLAGS += -DDBG_TRACE_CACHE
DBG_TRACE_FLAGS += -DDBG_TRACE_AFU
DBG_TRACE_FLAGS += -DDBG_TRACE_GBAR
DBG_TRACE_FLAGS += -DDBG_TRACE_TCU

# Control logic analyzer monitors
DBG_SCOPE_FLAGS += -DDBG_SCOPE_AFU
DBG_SCOPE_FLAGS += -DDBG_SCOPE_ISSUE
DBG_SCOPE_FLAGS += -DDBG_SCOPE_FETCH
DBG_SCOPE_FLAGS += -DDBG_SCOPE_LSU

ifdef NUM_CORES
# cluster configuration
CONFIGS_1c  := -DVX_CFG_NUM_CLUSTERS=1 -DVX_CFG_NUM_CORES=1
CONFIGS_2c  := -DVX_CFG_NUM_CLUSTERS=1 -DVX_CFG_NUM_CORES=2
CONFIGS_4c  := -DVX_CFG_NUM_CLUSTERS=1 -DVX_CFG_NUM_CORES=4  -DVX_CFG_L2_ENABLE
CONFIGS_8c	:= -DVX_CFG_NUM_CLUSTERS=1 -DVX_CFG_NUM_CORES=8  -DVX_CFG_L2_ENABLE
CONFIGS_16c	:= -DVX_CFG_NUM_CLUSTERS=1 -DVX_CFG_NUM_CORES=16 -DVX_CFG_L2_ENABLE
CONFIGS_32c := -DVX_CFG_NUM_CLUSTERS=2 -DVX_CFG_NUM_CORES=16 -DVX_CFG_L2_ENABLE
CONFIGS_64c := -DVX_CFG_NUM_CLUSTERS=4 -DVX_CFG_NUM_CORES=16 -DVX_CFG_L2_ENABLE
CONFIGS += $(CONFIGS_$(NUM_CORES)c)
endif

CONFIGS += -DSYNTHESIS -DASIC -DSYNOPSYS


# include paths
RTL_INCLUDE = -I$(ROOT_DIR)/sw -I$(ROOT_DIR)/hw -I$(RTL_DIR) -I$(RTL_DIR)/libs -I$(RTL_DIR)/interfaces -I$(RTL_DIR)/core -I$(RTL_DIR)/mem -I$(RTL_DIR)/cache
ifneq (,$(filter -DVX_CFG_EXT_F_ENABLE, $(XCONFIGS)))
	RTL_INCLUDE += -I$(RTL_DIR)/fpu
endif

RTL_INCLUDE += -I$(LIB_DIR)/no_mem

# Add TCU extension sources
ifneq (,$(filter -DVX_CFG_EXT_TCU_ENABLE, $(XCONFIGS)))
	RTL_PKGS += $(RTL_DIR)/tcu/VX_tcu_pkg.sv
	RTL_INCLUDE += -I$(RTL_DIR)/tcu
	ifneq (,$(filter -DVX_CFG_TCU_TYPE_DPI, $(XCONFIGS)))
		RTL_INCLUDE += -I$(RTL_DIR)/tcu/dpi
	endif
	ifneq (,$(filter -DVX_CFG_TCU_TYPE_DSP, $(XCONFIGS)))
		RTL_INCLUDE += -I$(RTL_DIR)/tcu/dsp
	endif
	ifneq (,$(filter -DVX_CFG_TCU_TYPE_BHF, $(XCONFIGS)))
		RTL_INCLUDE += -I$(RTL_DIR)/tcu/bhf
		RTL_INCLUDE += -J$(THIRD_PARTY_DIR)/hardfloat/source/RISCV
		RTL_INCLUDE += -I$(THIRD_PARTY_DIR)/hardfloat/source
	endif
	ifneq (,$(filter -DVX_CFG_TCU_TYPE_TFR, $(XCONFIGS)))
		RTL_INCLUDE += -I$(RTL_DIR)/tcu/tfr
	endif
endif

# Add DXA extension sources
ifneq (,$(filter -DVX_CFG_EXT_DXA_ENABLE, $(XCONFIGS)))
	RTL_PKGS += $(RTL_DIR)/dxa/VX_dxa_pkg.sv
	RTL_INCLUDE += -I$(RTL_DIR)/dxa
endif

# Add V extension sources
ifneq (,$(filter -DVX_CFG_EXT_V_ENABLE, $(XCONFIGS)))
	RTL_PKGS += $(RTL_DIR)/vpu/VX_vpu_pkg.sv
	RTL_INCLUDE += -I$(RTL_DIR)/vpu
endif

# Debugging
ifdef DEBUG
	CFLAGS += $(DBG_TRACE_FLAGS)
else
	CFLAGS += -DNDEBUG
endif

# Enable scope analyzer
ifdef SCOPE
	CFLAGS += -DSCOPE $(DBG_SCOPE_FLAGS)
	SCOPE_JSON += $(BUILD_DIR)/scope.json
endif

# Enable perf counters
ifdef PERF
	CFLAGS += -DPERF_ENABLE
endif

CFLAGS += -DVX_CFG_XLEN=$(XLEN) -DVX_CFG_XLEN_$(XLEN)
CFLAGS += $(CONFIGS)
CFLAGS += $(RTL_INCLUDE)

.PHONY: clean gen-sources synthesis-nosram synthesis-estsram synthesis

all: synthesis

gen-sources: $(BUILD_DIR)/sources.txt
$(BUILD_DIR)/sources.txt:
	$(SCRIPT_DIR)/gen_sources.sh -P $(CFLAGS) -C$(BUILD_DIR)/src -O$(BUILD_DIR)/sources.txt

synthesis: $(BUILD_DIR)/sources.txt
	cd $(BUILD_DIR); \
	TOP=$(TOP_LEVEL_ENTITY) \
	CLOCK_FREQ=$(CLOCK_FREQ) \
	DELAY_UNC=$(DELAY_UNC) \
	DELAY_IO=$(DELAY_IO) \
	SRC_FILE=sources.txt \
	SDC_FILE=$(SRC_DIR)/project.sdc \
	MEM_LIBS="$(SRAM_DB)" \
	SAIF_FILE="$(SAIF_FILE)" \
	SAIF_INST="$(SAIF_INST)" \
	OPT_LEVEL=$(OPT_LEVEL) \
	$(SYN_LIB_ENV) \
	TOOL_DIR=$(SCRIPT_DIR) \
	WALL_IGNORE=$(WALL_IGNORE) \
	dc_shell -f $(SRC_DIR)/project.tcl

synthesis-nosram: $(BUILD_DIR)/sources.txt
	cd $(BUILD_DIR); \
	TOP=$(TOP_LEVEL_ENTITY) \
	CLOCK_FREQ=$(CLOCK_FREQ) \
	DELAY_UNC=$(DELAY_UNC) \
	DELAY_IO=$(DELAY_IO) \
	SRC_FILE=sources.txt \
	SDC_FILE=$(SRC_DIR)/project.sdc \
	SAIF_FILE="$(SAIF_FILE)" \
	SAIF_INST="$(SAIF_INST)" \
	OPT_LEVEL=$(OPT_LEVEL) \
	$(SYN_LIB_ENV) \
	TOOL_DIR=$(SCRIPT_DIR) \
	WALL_IGNORE=$(WALL_IGNORE) \
	dc_shell -f $(SRC_DIR)/project.tcl

synthesis-estsram: $(BUILD_DIR)/sources.txt
	cd $(BUILD_DIR); \
	TOP=$(TOP_LEVEL_ENTITY) \
	CLOCK_FREQ=$(CLOCK_FREQ) \
	DELAY_UNC=$(DELAY_UNC) \
	DELAY_IO=$(DELAY_IO) \
	SRC_FILE=sources.txt \
	SDC_FILE=$(SRC_DIR)/project.sdc \
	BB_MODULES="VX_dp_ram_asic,VX_sp_ram_asic" \
	SAIF_FILE="$(SAIF_FILE)" \
	SAIF_INST="$(SAIF_INST)" \
	OPT_LEVEL=$(OPT_LEVEL) \
	$(SYN_LIB_ENV) \
	TOOL_DIR=$(SCRIPT_DIR) \
	WALL_IGNORE=$(WALL_IGNORE) \
	dc_shell -f $(SRC_DIR)/project.tcl

clean:
	$(RMDIR) $(BUILD_DIR)
