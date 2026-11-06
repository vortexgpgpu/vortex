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
