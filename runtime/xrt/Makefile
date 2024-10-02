include ../common.mk

TARGET ?= xrtsim

DESTDIR ?= $(CURDIR)/..

SRC_DIR := $(VORTEX_HOME)/runtime/xrt

CXXFLAGS += -std=c++14 -Wall -Wextra -Wfatal-errors
CXXFLAGS += -I$(INC_DIR) -I$(COMMON_DIR) -I$(ROOT_DIR)/hw -I$(XILINX_XRT)/include -I$(SIM_DIR)/common
CXXFLAGS += -fPIC

LDFLAGS += -shared -pthread
LDFLAGS += -L$(XILINX_XRT)/lib

SRCS := $(SRC_DIR)/vortex.cpp $(SIM_DIR)/common/util.cpp

# set up target types
ifeq ($(TARGET), xrtsim)
	XRTSIM = $(DESTDIR)/libxrtsim.so
	CXXFLAGS += -DXRTSIM -I$(SIM_DIR)/xrtsim
	LDFLAGS += -L$(DESTDIR) -lxrtsim
else
	LDFLAGS += -luuid -lxrt_coreutil
endif

PROJECT := libvortex-xrt.so

# Debugging
ifdef DEBUG
	CXXFLAGS += -g -O0
else
	CXXFLAGS += -O2 -DNDEBUG
endif

# Enable scope logic analyzer
ifdef SCOPE
	CXXFLAGS += -DSCOPE
	SRCS += $(COMMON_DIR)/scope.cpp
endif

all: $(DESTDIR)/$(PROJECT)

driver: $(DESTDIR)/libxrtsim.so

$(DESTDIR)/libxrtsim.so:
	DESTDIR=$(DESTDIR) $(MAKE) -C $(ROOT_DIR)/sim/xrtsim $(DESTDIR)/libxrtsim.so

$(DESTDIR)/$(PROJECT): $(SRCS) $(XRTSIM)
	$(CXX) $(CXXFLAGS) $(SRCS) $(LDFLAGS) -o $@

clean-driver:
	DESTDIR=$(DESTDIR) $(MAKE) -C $(ROOT_DIR)/sim/xrtsim clean

clean-runtime:
	rm -f $(DESTDIR)/$(PROJECT)

clean: clean-driver clean-runtime

.PHONY: all driver clean-driver clean-runtime clean