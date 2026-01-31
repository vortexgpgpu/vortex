include ../common.mk

DESTDIR ?= $(CURDIR)/..

CONFIG_FILE := $(DESTDIR)/rtlsim_rt_config.stamp
SRC_DIR := $(VORTEX_HOME)/runtime/rtlsim

CXXFLAGS += -std=c++17 -Wall -Wextra -pedantic -Wfatal-errors
CXXFLAGS += -I$(INC_DIR) -I$(ROOT_DIR)/hw -I$(SIM_DIR)/rtlsim -I$(SW_COMMON_DIR) -I$(RT_COMMON_DIR)
CXXFLAGS += -DXLEN_$(XLEN)

# Position independent code
CXXFLAGS += -fPIC

# Add external configuration
CXXFLAGS += $(CONFIGS)

LDFLAGS += -shared -pthread
LDFLAGS += -L$(DESTDIR) -lrtlsim

SRCS := $(SRC_DIR)/vortex.cpp

# Debugging
ifdef DEBUG
	CXXFLAGS += -g -O0
else
	CXXFLAGS += -O2 -DNDEBUG
endif

PROJECT := libvortex-rtlsim.so

.PHONY: all force driver clean-driver clean-runtime clean

all: $(DESTDIR)/$(PROJECT)

driver: $(DESTDIR)/librtlsim.so

$(DESTDIR)/librtlsim.so: force
	DESTDIR=$(DESTDIR) $(MAKE) -C $(ROOT_DIR)/sim/rtlsim $(DESTDIR)/librtlsim.so

$(DESTDIR)/$(PROJECT): $(SRCS) $(DESTDIR)/librtlsim.so $(CONFIG_FILE)
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
	DESTDIR=$(DESTDIR) $(MAKE) -C $(ROOT_DIR)/sim/rtlsim clean-lib

clean-runtime:
	rm -f $(DESTDIR)/$(PROJECT)

clean: clean-driver clean-runtime