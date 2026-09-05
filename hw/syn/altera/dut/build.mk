# Per-DUT build makefile for the Altera/Quartus flow.
#
# Copied into <PREFIX>_<family>_<dut>/ by the dispatcher and run there with
# DUT=<name>, so this file plays the role the former dut/<name>/Makefile did.
#
# Ordering mirrors the old per-DUT Makefiles: PROJECT/CONFIGS -> common.mk
# (computes XCONFIGS, RTL_DIR, IP_CACHE_DIR, ...) -> RTL_INCLUDE/RTL_PKGS.

ifndef DUT
$(error DUT is not set; run this through dut/Makefile, e.g. `make -C dut cache`)
endif

include ../catalog.mk

ifeq ($(filter $(DUT),$(DUTS)),)
$(error unknown DUT '$(DUT)'; see DUTS in dut/catalog.mk)
endif

PROJECT           = $($(DUT)_PRJ)
TOP_LEVEL_ENTITY  = $(PROJECT)
SRC_FILE          = $(PROJECT).sv
override CONFIGS += $($(DUT)_CFG)

include ../common.mk

RTL_INCLUDE = $($(DUT)_INC)
RTL_PKGS   += $($(DUT)_PKG)
