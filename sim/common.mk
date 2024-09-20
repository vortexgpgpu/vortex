ROOT_DIR := $(realpath ../..)
include $(ROOT_DIR)/config.mk

HW_DIR := $(VORTEX_HOME)/hw
RTL_DIR := $(HW_DIR)/rtl
DPI_DIR := $(HW_DIR)/dpi
SCRIPT_DIR := $(HW_DIR)/scripts
COMMON_DIR := $(VORTEX_HOME)/sim/common