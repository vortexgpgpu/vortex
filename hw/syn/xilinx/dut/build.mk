# Per-DUT build makefile for the Xilinx flow.
#
# Copied into <PREFIX>_<dut>/ by the dispatcher and run there with DUT=<name>,
# so this file plays the role the former dut/<name>/Makefile did. Everything it
# needs comes from catalog.mk; nothing here is DUT-specific.
#
# Ordering matters and mirrors the old per-DUT Makefiles exactly:
#   PROJECT/CONFIGS  ->  common.mk (computes XCONFIGS, RTL_DIR, ...)  ->
#   RTL_INCLUDE/RTL_PKGS (which read XCONFIGS)  ->  extensions.mk

ifndef DUT
$(error DUT is not set; run this through dut/Makefile, e.g. `make -C dut tcu`)
endif

include ../catalog.mk

ifeq ($(filter $(DUT),$(DUTS)),)
$(error unknown DUT '$(DUT)'; see DUTS in dut/catalog.mk)
endif

PROJECT           = $($(DUT)_PRJ)
TOP_LEVEL_ENTITY  = $(PROJECT)
SRC_FILE          = $(PROJECT).sv
FPU_IP            = $($(DUT)_IP)
override CONFIGS += $($(DUT)_CFG)

include ../common.mk

RTL_INCLUDE = $($(DUT)_INC)
RTL_PKGS   += $($(DUT)_PKG)

ifeq ($($(DUT)_EXT),1)
include $(VORTEX_HOME)/hw/syn/extensions.mk
endif

# An AFU build reads the platform tops too, which declare one AXI master per
# GPU memory port literally: resolve the count from the elaborated
# configuration, as the xrt flow does.
ifeq ($($(DUT)_AFU),1)
AFU_MEM_PORTS := $(shell $(SCRIPT_DIR)/afu_mem_ports.sh -c .afu_mem_ports $(VERILATOR_PATH)/bin/verilator $(filter-out -J% -DSYNTHESIS -DVIVADO,$(CFLAGS)) $(RTL_PKGS))
ifeq ($(AFU_MEM_PORTS),)
$(error cannot resolve the AFU memory port count for this configuration)
endif
CFLAGS += -DAFU_MEM_PORTS=$(AFU_MEM_PORTS)
endif
