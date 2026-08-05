ROOT_DIR := $(realpath ../../..)
include $(ROOT_DIR)/config.mk

CONFIGS := $(if $(findstring -DVX_CFG_EXT_DTCU_ENABLE,$(CONFIGS)),$(CONFIGS),$(CONFIGS) -DVX_CFG_EXT_DTCU_ENABLE)
# Atomics are the mechanism under test, not an optimization: dtensor_check() and every
# handshake word here are AMOs, which take the cache's probe-and-invalidate path and
# resolve at the LLC. A plain load caches the value locally and spins on it forever.
CONFIGS := $(if $(findstring -DVX_CFG_EXT_A_ENABLE,$(CONFIGS)),$(CONFIGS),$(CONFIGS) -DVX_CFG_EXT_A_ENABLE)
# L2 is REQUIRED: it is where the engine's completion store and the consumer's AMO meet.
# Without it the dcache is the LLC, the DTCU bypasses it, and the consumer never sees
# the flag.
CONFIGS += -DVX_CFG_L2_ENABLE

# The whole point of this test is that the observer is a DIFFERENT core than the
# submitter. 4 cores at SOCKET_SIZE=2 gives 2 sockets x 2 cores, which covers BOTH
# consumer positions in one run:
#   (a) a different core in the SAME socket -- for DTCU_socket, this one hits the very
#       dcache the engine wrote D into;
#   (b) a core in a DIFFERENT socket -- which has to fall through to L2 instead.
# Those are different coherence stories and both have to work.
CONFIGS += -DVX_CFG_NUM_CLUSTERS=1
CONFIGS += -DVX_CFG_NUM_CORES=4
CONFIGS += -DVX_CFG_SOCKET_SIZE=2

PROJECT := dtcu_xcore

SRC_DIR := $(VORTEX_HOME)/tests/regression/$(PROJECT)

SRCS := $(SRC_DIR)/main.cpp $(SW_COMMON_DIR)/rvfloats.cpp $(SW_COMMON_DIR)/softfloat_ext.cpp

VX_SRCS := $(SRC_DIR)/kernel.cpp

CXXFLAGS += -I$(THIRD_PARTY_DIR)/softfloat/source/include

LDFLAGS += $(THIRD_PARTY_DIR)/softfloat/build/Linux-x86_64-GCC/softfloat.a

KERNEL_LIB := vortex2

include ../common.mk
