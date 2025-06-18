include ../common.mk

DESTDIR ?= $(CURDIR)/..

CONFIG_FILE := $(DESTDIR)/simx_rt_config.stamp
SRC_DIR := $(VORTEX_HOME)/runtime/simx

CXXFLAGS += -std=c++17 -Wall -Wextra -Wfatal-errors
CXXFLAGS += -fPIC -Wno-maybe-uninitialized
CXXFLAGS += -I$(INC_DIR) -I../common -I$(ROOT_DIR)/hw -I$(SIM_DIR)/simx -I$(SW_COMMON_DIR) -I$(RT_COMMON_DIR)
CXXFLAGS += -DXLEN_$(XLEN)
CXXFLAGS += $(CONFIGS)

LDFLAGS += -shared -pthread
LDFLAGS += -L$(DESTDIR) -lsimx

SRCS := $(SRC_DIR)/vortex.cpp

# Debugging
ifdef DEBUG
	CXXFLAGS += -g -O0
else
	CXXFLAGS += -O2 -DNDEBUG
endif

PROJECT := libvortex-simx.so

.PHONY: all force driver clean-driver clean-runtime clean

all: $(DESTDIR)/$(PROJECT)

driver: $(DESTDIR)/libsimx.so

$(DESTDIR)/libsimx.so: force
	DESTDIR=$(DESTDIR) $(MAKE) -C $(ROOT_DIR)/sim/simx $(DESTDIR)/libsimx.so

$(DESTDIR)/$(PROJECT): $(SRCS) $(DESTDIR)/libsimx.so $(CONFIG_FILE)
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
	DESTDIR=$(DESTDIR) $(MAKE) -C $(ROOT_DIR)/sim/simx clean-lib

clean-runtime:
	rm -f $(DESTDIR)/$(PROJECT)

clean: clean-driver clean-runtime