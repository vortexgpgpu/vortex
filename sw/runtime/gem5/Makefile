include ../common.mk

# HOST_ARCH selects the cross-compiler for the simulated host ISA
# inside gem5 (see docs/proposals/gem5_simx_v3_proposal.md §3.5).
# Default x86_64 has no toolchain install requirement; aarch64/armhf
# need ci/gem5_install.sh to have run sudo-apt for the cross-compilers.
HOST_ARCH ?= x86_64

DESTDIR ?= $(CURDIR)/..

SRC_DIR := $(VORTEX_HOME)/sw/runtime/gem5

CXXFLAGS += -std=c++17 -Wall -Wextra -pedantic -Wfatal-errors -Werror
CXXFLAGS += -I$(INC_DIR) -I$(ROOT_DIR)/sw -I$(ROOT_DIR)/hw -I$(DESTDIR) -I$(SW_COMMON_DIR) -I$(RT_COMMON_DIR)
CXXFLAGS += -DXLEN_$(XLEN)
CXXFLAGS += -fPIC
CXXFLAGS += $(CONFIGS)

# OPAE-shaped MMIO constants come from the generated vortex_opae.h
# at build/sw/ (already on the include path via -I$(ROOT_DIR)/sw).
# vortex.cpp does `#include <vortex_opae.h>` for the AFU_IMAGE_*
# defines. Unlike sw/runtime/opae/Makefile we do NOT call
# afu_json_mgr — configure already generated the header from
# vortex_opae.toml at build time.

# Per-arch compiler selection. The cross-compilers are sysroot-aware
# (Ubuntu's gcc-aarch64-linux-gnu ships the matching libstdc++); no
# extra --sysroot flags needed.
#
# Cross-compiled outputs land in $(DESTDIR)/$(HOST_ARCH)/ alongside
# the stub's libvortex.so (also cross-compiled). The simulated ARM
# process's LD_LIBRARY_PATH points at that one dir to find both.
ifeq ($(HOST_ARCH),x86_64)
    CXX := g++
    ARCH_SUFFIX := x86_64
    OUT_DIR := $(DESTDIR)
else ifeq ($(HOST_ARCH),aarch64)
    CXX := aarch64-linux-gnu-g++
    ARCH_SUFFIX := aarch64
    OUT_DIR := $(DESTDIR)/aarch64
else ifeq ($(HOST_ARCH),armhf)
    CXX := arm-linux-gnueabihf-g++
    ARCH_SUFFIX := armhf
    OUT_DIR := $(DESTDIR)/armhf
else
    $(error HOST_ARCH must be one of: x86_64, aarch64, armhf (got $(HOST_ARCH)))
endif

LDFLAGS += -shared -pthread

SRCS = $(SRC_DIR)/vortex.cpp $(SRC_DIR)/driver.cpp $(RT_COMMON_DIR)/utils.cpp

# Debug / release
ifdef DEBUG
    CXXFLAGS += -g -O0
else
    CXXFLAGS += -O2 -DNDEBUG
endif

PROJECT := libvortex-gem5-$(ARCH_SUFFIX).so

.PHONY: all force clean

all: $(OUT_DIR)/$(PROJECT)

$(OUT_DIR)/$(PROJECT): $(SRCS)
	@mkdir -p $(OUT_DIR)
	$(CXX) $(CXXFLAGS) $(SRCS) $(LDFLAGS) -Wl,-soname,$(PROJECT) -o $@

clean:
	rm -f $(DESTDIR)/libvortex-gem5-*.so
	rm -f $(DESTDIR)/aarch64/libvortex-gem5-*.so
	rm -f $(DESTDIR)/armhf/libvortex-gem5-*.so
