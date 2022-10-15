TARGET ?= opaesim

OPAESIM_DIR = ../../sim/opaesim

RTL_DIR=../../hw/rtl

SCRIPT_DIR=../../hw/scripts

CXXFLAGS += -std=c++11 -Wall -Wextra -pedantic -Wfatal-errors

CXXFLAGS += -I. -I../include -I../common/ -I../../hw -I$(OPAESIM_DIR)

# Position independent code
CXXFLAGS += -fPIC

# Add external configuration
CXXFLAGS += $(CONFIGS)

# Dump perf stats
CXXFLAGS += -DDUMP_PERF_STATS

LDFLAGS += -shared -luuid -ldl -pthread

SRCS = vortex.cpp api_init.cpp ../common/vx_utils.cpp

# Debugigng
ifdef DEBUG
	CXXFLAGS += -g -O0
else    
	CXXFLAGS += -O2 -DNDEBUG
endif

# Enable scope analyzer
ifdef SCOPE
	CXXFLAGS += -DSCOPE	
	SRCS += scope.cpp
	SCOPE_H = scope-defs.h 
endif

# Enable perf counters
ifdef PERF
	CXXFLAGS += -DPERF_ENABLE
endif

PROJECT = libvortex.so

all: $(PROJECT)

scope-defs.h: $(SCRIPT_DIR)/scope.json
	$(SCRIPT_DIR)/scope.py $(CONFIGS) -D NOGLOBALS -cc scope-defs.h -vl $(RTL_DIR)/scope-defs.vh $(SCRIPT_DIR)/scope.json

# generate scope data
scope: scope-defs.h

$(PROJECT): $(SRCS) $(SCOPE_H)
ifeq ($(TARGET), opaesim)
	$(MAKE) -C $(OPAESIM_DIR)
endif
	$(CXX) $(CXXFLAGS) $(SRCS) $(LDFLAGS) -o $(PROJECT)

clean:
	rm -rf $(PROJECT) *.o scope-defs.h
ifeq ($(TARGET), opaesim)
	$(MAKE) -C $(OPAESIM_DIR) clean
endif