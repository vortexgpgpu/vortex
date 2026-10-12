include ../common.mk

# TARGET=hw runs a maxsim image (hw/syn/maxsim) on the V80 through VRT;
# TARGET=avedsim simulates the same kernel in-process (sim/maxsim).
ifeq ($(strip $(TARGET)),)
override TARGET := avedsim
endif

DESTDIR ?= $(CURDIR)/..

CONFIG_FILE := $(DESTDIR)/maxsim_rt_config.stamp
SRC_DIR := $(VORTEX_HOME)/sw/runtime/maxsim

CXXFLAGS += -std=c++17 -Wall -Wextra -Wfatal-errors -Werror
CXXFLAGS += -I$(INC_DIR) -I$(ROOT_DIR)/sw -I$(ROOT_DIR)/hw -I$(SIM_COMMON_DIR) -I$(SW_COMMON_DIR) -I$(RT_COMMON_DIR)
CXXFLAGS += -DVX_CFG_XLEN=$(XLEN) -DVX_CFG_XLEN_$(XLEN)
CXXFLAGS += -fPIC
CXXFLAGS += $(CONFIGS)

LDFLAGS += -shared -pthread -Wl,-rpath,'$$ORIGIN'

SRCS := $(SRC_DIR)/vortex.cpp $(RT_COMMON_DIR)/utils.cpp $(SIM_COMMON_DIR)/cmd_processor.cpp

ifeq ($(TARGET), avedsim)
CXXFLAGS += -DMAXSIM_EMUL -I$(SIM_DIR)/maxsim
LDFLAGS += -L$(DESTDIR) -lmaxsim_emul
EMUL := $(DESTDIR)/libmaxsim_emul.so
else ifeq ($(TARGET), hw)
# VRT's public headers include <json/json.h> and <libxml/parser.h> from
# versioned subdirectories.
CXXFLAGS += $(if $(VRT_HOME),-I$(VRT_HOME)/include) $(shell pkg-config --cflags jsoncpp libxml-2.0 2>/dev/null)
LDFLAGS += $(if $(VRT_HOME),-L$(VRT_HOME)/lib) -lvrt -lvrtdpp
else
$(error TARGET must be hw or avedsim (got '$(TARGET)'))
endif

ifdef DEBUG
	CXXFLAGS += -g -O0
else
	CXXFLAGS += -O2 -DNDEBUG
endif

PROJECT := libvortex-maxsim.so

.PHONY: all force driver clean-driver clean-runtime clean

all: $(DESTDIR)/$(PROJECT)

driver: $(EMUL)

$(DESTDIR)/libmaxsim_emul.so: force
	DESTDIR=$(DESTDIR) $(MAKE) -C $(ROOT_DIR)/sim/maxsim lib CONFIGS='$(CONFIGS)' $(if $(PERF),PERF=$(PERF))

$(DESTDIR)/$(PROJECT): $(SRCS) $(EMUL) $(CONFIG_FILE)
	$(CXX) $(CXXFLAGS) $(SRCS) $(LDFLAGS) -o $@

$(CONFIG_FILE): force
	@mkdir -p $(@D)
	@TMP=$@.tmp.$$$$ ; \
	 printf '%s\n' "$(CXXFLAGS)" "$(TARGET)" > $$TMP ; \
	 if ! cmp -s $$TMP $@ 2>/dev/null; then mv -f $$TMP $@ ; else rm -f $$TMP ; fi

clean-driver:
	DESTDIR=$(DESTDIR) $(MAKE) -C $(ROOT_DIR)/sim/maxsim clean

clean-runtime:
	rm -f $(DESTDIR)/$(PROJECT)

clean: clean-driver clean-runtime
