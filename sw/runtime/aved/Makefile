include ../common.mk

# hw / sim / emu all link real VRT and differ only in the device binary;
# avedsim links the in-process model instead.
TARGET ?= avedsim

DESTDIR ?= $(CURDIR)/..

CONFIG_FILE := $(DESTDIR)/aved_config.stamp
SRC_DIR := $(VORTEX_HOME)/sw/runtime/aved

CXXFLAGS += -std=c++17 -Wall -Wextra -Wfatal-errors -Werror
CXXFLAGS += -I$(INC_DIR) -I$(ROOT_DIR)/sw -I$(ROOT_DIR)/hw -I$(SW_COMMON_DIR) -I$(RT_COMMON_DIR)
CXXFLAGS += -DVX_CFG_XLEN=$(XLEN) -DVX_CFG_XLEN_$(XLEN)
CXXFLAGS += -fPIC

LDFLAGS += -shared -pthread

SRCS := $(SRC_DIR)/vortex.cpp $(SW_COMMON_DIR)/util.cpp

ifeq ($(TARGET), emu)
$(error TARGET=emu is unavailable: the V80 emulation platform is a C-model \
        that does not accept RTL kernels. Use TARGET=avedsim for fast \
        functional iteration, or --driver=simx for the C-model)
endif

# set up target types
ifeq ($(TARGET), avedsim)
	AVEDSIM = $(DESTDIR)/libavedsim.so
	CXXFLAGS += -DAVEDSIM -I$(SIM_DIR)/avedsim
	LDFLAGS += -L$(DESTDIR) -lavedsim
else
# Packaged VRT (libvrt-dev) installs to the default include and library paths;
# VRT_HOME only matters for a hand-built VRT in a custom prefix.
	CXXFLAGS += -I$(VRT_HOME)/include
	LDFLAGS += -L$(VRT_HOME)/lib -lvrt
# VRT's public headers include <json/json.h> and <libxml/parser.h>
# transitively, and distributions put those under versioned subdirectories.
# Ask pkg-config rather than hardcoding a path -- this is what the CPATH
# include-shim under ~/dev/v80/inc-shim used to paper over.
	CXXFLAGS += $(shell pkg-config --cflags jsoncpp libxml-2.0 2>/dev/null)
endif

PROJECT := libvortex-aved.so

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

driver: $(DESTDIR)/libavedsim.so

$(DESTDIR)/libavedsim.so: force
	DESTDIR=$(DESTDIR) $(MAKE) -C $(ROOT_DIR)/sim/avedsim $(DESTDIR)/libavedsim.so

$(DESTDIR)/$(PROJECT): $(SRCS) $(AVEDSIM) $(CONFIG_FILE)
	$(CXX) $(CXXFLAGS) $(SRCS) $(LDFLAGS) -o $@

# updates the timestamp when flags changed.
# Use a PID-suffixed tmp file and tolerate concurrent invocations
# (blackbox.sh + tests/regression/*/Makefile both invoke `make -C` here).
$(CONFIG_FILE): force
	@mkdir -p $(@D)
	@TMP=$@.tmp.$$$$ ; \
	 printf '%s\n' "$(CXXFLAGS)" > $$TMP ; \
	 if ! cmp -s $$TMP $@ 2>/dev/null; then \
	   mv -f $$TMP $@ ; \
	 else \
	   rm -f $$TMP ; \
	 fi

clean-driver:
	DESTDIR=$(DESTDIR) $(MAKE) -C $(ROOT_DIR)/sim/avedsim clean

clean-runtime:
	rm -f $(DESTDIR)/$(PROJECT)

clean: clean-driver clean-runtime
