OPAE_HOME ?= /tools/opae/1.4.0

CXXFLAGS += -std=c++11 -O2 -DNDEBUG -Wall -Wextra -pedantic -Wfatal-errors
#CXXFLAGS += -std=c++11 -O0 -g -Wall -Wextra -pedantic -Wfatal-errors

CXXFLAGS += -I../include -I$(OPAE_HOME)/include  -I../../hw

LDFLAGS += -L$(OPAE_HOME)/lib -luuid -lopae-c

#SCOPE=1

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

RTL_DIR=../../hw/rtl

SCRIPT_DIR=../../hw/scripts

PROJECT = libvortex.so

AFU_JSON_INFO = vortex_afu.h

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

# AFU info from JSON file, including AFU UUID
json: ../../hw/opae/vortex_afu.json
	afu_json_mgr json-info --afu-json=$^ --c-hdr=$@

scope-defs.h: $(SCRIPT_DIR)/scope.json
	$(SCRIPT_DIR)/scope.py $(CONFIGS) -cc scope-defs.h -vl $(RTL_DIR)/scope-defs.vh $(SCRIPT_DIR)/scope.json

# generate scope data
scope: scope-defs.h

$(PROJECT): $(SRCS) $(SCOPE_H)
	$(CXX) $(CXXFLAGS) -DUSE_FPGA $^ $(LDFLAGS) -o $(PROJECT)

.depend: $(SRCS)
	$(CXX) $(CXXFLAGS) -MM $(SRCS) > .depend;

clean:
	rm -rf $(PROJECT) *.o .depend

ifneq ($(MAKECMDGOALS),clean)
    -include .depend
endif
