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

# The card's FireSim platform, from its descriptor: it decides which variables
# name the image and the card (those of the card's native driver).
FIRESIM_CARD ?= u55c
FIRESIM_PLATFORM := $(shell python3 $(VORTEX_HOME)/hw/scripts/gen_platform.py --card=$(FIRESIM_CARD) --emit=firesim-mk | sed -n 's/^PLATFORM := //p')
ifeq ($(FIRESIM_PLATFORM),)
$(error cannot resolve the FireSim platform of card '$(FIRESIM_CARD)')
endif
ifeq ($(FIRESIM_PLATFORM), aved)
CXXFLAGS += -DFIRESIM_PLATFORM_AVED
endif

# TARGET=avedsim runs the metasimulator in-process in place of the card: the
# same FireSim-transformed target in Verilator, reached through the emulated
# host interface, with no image and no VRT. It is the aved counterpart of the
# native driver's avedsim.
ifeq ($(TARGET), avedsim)
CXXFLAGS += -DFIRESIM_EMUL
endif

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
	DESTDIR=$(DESTDIR) $(MAKE) -C $(ROOT_DIR)/sim/firesim lib FIRESIM_CARD=$(FIRESIM_CARD) TARGET=$(TARGET)

$(DESTDIR)/$(PROJECT): $(SRCS) $(DESTDIR)/libfiresim.so $(CONFIG_FILE)
	$(CXX) $(CXXFLAGS) $(SRCS) $(LDFLAGS) -o $@

# Forced, as in every other runtime: without a prerequisite the stamp is written
# once and never compared again, so a changed CONFIGS keeps linking the runtime
# built for the previous one.
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
	DESTDIR=$(DESTDIR) $(MAKE) -C $(ROOT_DIR)/sim/firesim clean

clean-runtime:
	rm -f $(DESTDIR)/$(PROJECT) $(CONFIG_FILE)

clean: clean-driver clean-runtime
