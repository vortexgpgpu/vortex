OPAE_HOME ?= /tools/opae/1.4.0

RTL_DIR=../../hw/rtl

SCRIPT_DIR=../../hw/scripts

OPAE_SYN_DIR=../../hw/syn/opae

CXXFLAGS += -std=c++11 -O2 -DNDEBUG -Wall -Wextra -pedantic -Wfatal-errors
#CXXFLAGS += -std=c++11 -O0 -g -Wall -Wextra -pedantic -Wfatal-errors

CXXFLAGS += -I. -I../include -I../../hw -I$(OPAE_HOME)/include -I$(OPAE_SYN_DIR)

LDFLAGS += -L$(OPAE_HOME)/lib -luuid -lopae-c-ase

# stack execution protection
LDFLAGS +=-z noexecstack

# data relocation and projection
LDFLAGS +=-z relro -z now

# stack buffer overrun detection
CXXFLAGS +=-fstack-protector

# Position independent code
CXXFLAGS += -fPIC

# Add external configuration
CXXFLAGS += $(CONFIGS)

# Dump perf stats
CXXFLAGS += -DDUMP_PERF_STATS

LDFLAGS += -shared

PROJECT = libvortex.so

SRCS = ../common/opae.cpp ../common/vx_utils.cpp

# Enable scope analyzer
ifdef SCOPE
	CXXFLAGS += -DSCOPE	
	SRCS += ../common/vx_scope.cpp
	SCOPE_H = scope-defs.h 
endif

# Enable perf counters
ifdef PERF
	CXXFLAGS += -DPERF_ENABLE
endif

all: $(PROJECT)

$(OPAE_SYN_DIR)/vortex_afu.h:
	$(MAKE) -C $(OPAE_SYN_DIR) vortex_afu.h

scope-defs.h: $(SCRIPT_DIR)/scope.json
	$(SCRIPT_DIR)/scope.py $(CONFIGS) -cc scope-defs.h -vl $(RTL_DIR)/scope-defs.vh $(SCRIPT_DIR)/scope.json

# generate scope data
scope: scope-defs.h	

$(PROJECT): $(SRCS) $(OPAE_SYN_DIR)/vortex_afu.h $(SCOPE_H)
	$(CXX) $(CXXFLAGS) -DUSE_ASE $(SRCS) $(LDFLAGS) -o $(PROJECT)

clean:
	rm -rf $(PROJECT) *.o scope-defs.h