VLSIM_DIR = ../../sim/vlsim

RTL_DIR=../../hw/rtl

SCRIPT_DIR=../../hw/scripts

CXXFLAGS += -std=c++11 -O2 -DNDEBUG -Wall -Wextra -pedantic -Wfatal-errors
#CXXFLAGS += -std=c++11 -O0 -g -Wall -Wextra -pedantic -Wfatal-errors

CXXFLAGS += -I. -I../include -I../../hw -I$(VLSIM_DIR)

LDFLAGS += $(VLSIM_DIR)/libopae-c-vlsim.a

# Position independent code
CXXFLAGS += -fPIC

# Add external configuration
CXXFLAGS += $(CONFIGS)

# Dump perf stats
CXXFLAGS += -DDUMP_PERF_STATS

LDFLAGS += -shared -pthread

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

PROJECT = libvortex.so

all: $(PROJECT)

scope-defs.h: $(SCRIPT_DIR)/scope.json
	$(SCRIPT_DIR)/scope.py $(CONFIGS) -cc scope-defs.h -vl $(RTL_DIR)/scope-defs.vh $(SCRIPT_DIR)/scope.json

# generate scope data
scope: scope-defs.h

$(PROJECT): $(SRCS) $(SCOPE_H)
	$(SCOPE_ENABLE) $(PERF_ENABLE) $(MAKE) -C $(VLSIM_DIR) static
	$(CXX) $(CXXFLAGS) -DUSE_VLSIM $(SRCS) $(LDFLAGS) -o $(PROJECT)

clean:
	$(MAKE) -C $(VLSIM_DIR) clean-static
	rm -rf $(PROJECT) *.o scope-defs.h