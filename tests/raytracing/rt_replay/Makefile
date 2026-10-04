ROOT_DIR := $(realpath ../../..)
include $(ROOT_DIR)/config.mk

# The scene format the RTU walks is a build-time choice (VX_CFG_RTU_BVH_WIDTH):
# replay against the configuration that recorded the log, so CONFIGS is used
# as given, plus the RTU itself.
CONFIGS := $(if $(findstring -DVX_CFG_EXT_RTU_ENABLE,$(CONFIGS)),$(CONFIGS),$(CONFIGS) -DVX_CFG_EXT_RTU_ENABLE)

PROJECT := rt_replay

SRC_DIR := $(VORTEX_HOME)/tests/raytracing/$(PROJECT)

SRCS := $(SRC_DIR)/main.cpp
HDRS := $(SRC_DIR)/common.h

VX_SRCS := $(SRC_DIR)/kernel.cpp
VX_HDRS := $(SRC_DIR)/common.h

# Usage (from the build dir), e.g. for the LumiBench RTU config on rtlsim:
#   CONFIGS="$(cat <rt_dir>/CONFIGS.txt) -DVX_DBG_STALL_TIMEOUT=2000000000" \
#     make -C tests/raytracing/rt_replay run-rtlsim OPTS="-f <raylog> -r 0:1024"
# The stall-timeout define is required on rtlsim: one hard ray can keep every
# warp waiting in vx_rt_wait longer than the scheduler watchdog's default.
# Logs come from SimX with VX_RTU_RAYLOG=<path>; ./rt_replay -h lists options.
OPTS ?=

KERNEL_LIB := vortex2

include ../common.mk
