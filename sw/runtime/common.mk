ROOT_DIR := $(realpath ../../..)
include $(ROOT_DIR)/config.mk

SIM_DIR := $(VORTEX_HOME)/sim
HW_DIR := $(VORTEX_HOME)/hw

INC_DIR := $(VORTEX_HOME)/sw/runtime/include
RT_COMMON_DIR := $(VORTEX_HOME)/sw/runtime/common

# Resolve the toml + CONFIGS overrides into the canonical -D... list, the
# same way sim/simx/Makefile does. Project the resolved hardware config
# as -DVX_CFG_* flags (all backends), so runtime code need not
# #include <VX_config.h>.
XCONFIGS := $(shell python3 $(ROOT_DIR)/ci/gen_config.py --config=$(VORTEX_HOME)/VX_config.toml --cflags='$(CONFIGS) -DVX_CFG_XLEN=$(XLEN)')
CXXFLAGS += $(XCONFIGS)