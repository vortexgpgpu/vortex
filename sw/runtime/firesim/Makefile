include ../common.mk

DESTDIR ?= $(CURDIR)/..

CONFIG_FILE := $(DESTDIR)/firesim_rt_config.stamp
SRC_DIR := $(VORTEX_HOME)/sw/runtime/firesim

CXXFLAGS += -std=c++17 -Wall -Wextra -pedantic -Wfatal-errors -Werror
CXXFLAGS += -I$(INC_DIR) -I$(ROOT_DIR)/sw -I$(ROOT_DIR)/hw -I$(SIM_DIR)/firesim -I$(SIM_COMMON_DIR) -I$(SW_COMMON_DIR) -I$(RT_COMMON_DIR)
CXXFLAGS += -DVX_CFG_XLEN=$(XLEN) -DVX_CFG_XLEN_$(XLEN)

# Position independent code
CXXFLAGS += -fPIC

# Add external configuration
CXXFLAGS += $(CONFIGS)

LDFLAGS += -shared -pthread
# Find libfiresim.so siblings at runtime in the same dir libvortex-firesim.so lives in.
LDFLAGS += -Wl,-rpath,'$$ORIGIN'
LDFLAGS += -L$(DESTDIR) -lfiresim

SRCS := $(SRC_DIR)/vortex.cpp $(RT_COMMON_DIR)/utils.cpp \
        $(SIM_COMMON_DIR)/cmd_processor.cpp

# Debugging
ifdef DEBUG
	CXXFLAGS += -g -O0
else
	CXXFLAGS += -O2 -DNDEBUG
endif

PROJECT := libvortex-firesim.so

.PHONY: all force driver clean-driver clean-runtime clean

all: $(DESTDIR)/$(PROJECT)

driver: $(DESTDIR)/libfiresim.so

$(DESTDIR)/libfiresim.so: force
	DESTDIR=$(DESTDIR) $(MAKE) -C $(ROOT_DIR)/sim/firesim lib

$(DESTDIR)/$(PROJECT): $(SRCS) $(DESTDIR)/libfiresim.so $(CONFIG_FILE)
	$(CXX) $(CXXFLAGS) $(SRCS) $(LDFLAGS) -o $@

$(CONFIG_FILE):
	echo "$(CXXFLAGS)" > $@.tmp
	@if ! cmp -s $@.tmp $@; then mv $@.tmp $@; else rm $@.tmp; fi

clean-driver:
	DESTDIR=$(DESTDIR) $(MAKE) -C $(ROOT_DIR)/sim/firesim clean

clean-runtime:
	rm -f $(DESTDIR)/$(PROJECT) $(CONFIG_FILE)

clean: clean-driver clean-runtime
