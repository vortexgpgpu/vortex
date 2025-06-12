include ../common.mk

TARGET ?= xrtsim

DESTDIR ?= $(CURDIR)/..

CONFIG_FILE := $(DESTDIR)/xrt_config.stamp
SRC_DIR := $(VORTEX_HOME)/runtime/xrt

CXXFLAGS += -std=c++17 -Wall -Wextra -Wfatal-errors
CXXFLAGS += -I$(INC_DIR) -I$(ROOT_DIR)/hw -I$(XILINX_XRT)/include -I$(SW_COMMON_DIR) -I$(RT_COMMON_DIR)
CXXFLAGS += -DXLEN_$(XLEN)
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
	SRCS += $(RT_COMMON_DIR)/scope.cpp
endif

# Enable ILA logic analyzer
ifdef CHIPSCOPE
	CXXFLAGS += -DCHIPSCOPE
endif

.PHONY: all force driver clean-driver clean-runtime clean

all: $(DESTDIR)/$(PROJECT)

driver: $(DESTDIR)/libxrtsim.so

$(DESTDIR)/libxrtsim.so: force
	DESTDIR=$(DESTDIR) $(MAKE) -C $(ROOT_DIR)/sim/xrtsim $(DESTDIR)/libxrtsim.so

$(DESTDIR)/$(PROJECT): $(SRCS) $(XRTSIM) $(CONFIG_FILE)
	$(CXX) $(CXXFLAGS) $(SRCS) $(LDFLAGS) -o $@

# updates the timestamp when flags changed.
$(CONFIG_FILE): force
	@mkdir -p $(@D)
	@printf '%s\n' "$(CXXFLAGS)" > $@.tmp
	@if ! cmp -s $@.tmp $@; then \
	  mv $@.tmp $@; \
	else \
	  rm $@.tmp; \
	fi

clean-driver:
	DESTDIR=$(DESTDIR) $(MAKE) -C $(ROOT_DIR)/sim/xrtsim clean

clean-runtime:
	rm -f $(DESTDIR)/$(PROJECT)

clean: clean-driver clean-runtime