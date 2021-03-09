OPAE_HOME ?= /tools/opae/1.4.0

CXXFLAGS += -std=c++11 -O2 -DNDEBUG -Wall -Wextra -pedantic -Wfatal-errors
#CXXFLAGS += -std=c++11 -O0 -g -Wall -Wextra -pedantic -Wfatal-errors

CXXFLAGS += -I../include -I$(OPAE_HOME)/include  -I../../hw

LDFLAGS += -L$(OPAE_HOME)/lib

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

FPGA_LIBS += -luuid -lopae-c

ASE_LIBS += -luuid -lopae-c-ase

VLSIM_LIBS += -lopae-c-vlsim

ASE_DIR = ase

VLSIM_DIR = vlsim

RTL_DIR=../../hw/rtl

SCRIPT_DIR=../../hw/scripts

PROJECT = libvortex.so

PROJECT_ASE = $(ASE_DIR)/libvortex.so

PROJECT_VLSIM = $(VLSIM_DIR)/libvortex.so

AFU_JSON_INFO = vortex_afu.h

SRCS = vortex.cpp ../common/vx_utils.cpp

# Enable scope analyzer
ifdef SCOPE
	CXXFLAGS += -DSCOPE	
	SRCS += vx_scope.cpp
	SCOPE_ENABLE = SCOPE=1
	SCOPE_H = scope-defs.h 
endif

# Enable perf counters
ifdef PERF
	CXXFLAGS += -DPERF_ENABLE
	PERF_ENABLE = PERF=1
endif

all: vlsim

# AFU info from JSON file, including AFU UUID
json: ../../hw/opae/vortex_afu.json
	afu_json_mgr json-info --afu-json=$^ --c-hdr=$@

scope-defs.h: $(SCRIPT_DIR)/scope.json
	$(SCRIPT_DIR)/scope.py $(RTL_INCLUDE) $(CONFIGS) -cc scope-defs.h -vl $(RTL_DIR)/scope-defs.vh $(SCRIPT_DIR)/scope.json

# generate scope data
scope: scope-defs.h

vlsim-hw: $(SCOPE_H)
	$(SCOPE_ENABLE) $(PERF_ENABLE) $(MAKE) -C vlsim

fpga: $(SRCS) $(SCOPE_H)
	$(CXX) $(CXXFLAGS) -DUSE_FPGA $^ $(LDFLAGS) $(FPGA_LIBS) -o $(PROJECT)

asesim: $(SRCS) $(ASE_DIR) $(SCOPE_H)
	$(CXX) $(CXXFLAGS) -DUSE_ASE $(SRCS) $(LDFLAGS) $(ASE_LIBS) -o $(PROJECT_ASE)

vlsim: $(SRCS) vlsim-hw
	$(CXX) $(CXXFLAGS) -DUSE_VLSIM $(SRCS) $(LDFLAGS) -L./vlsim $(VLSIM_LIBS) -o $(PROJECT_VLSIM)

vortex.o: vortex.cpp
	$(CXX) $(CXXFLAGS) -c vortex.cpp -o $@

$(ASE_DIR):
	mkdir -p ase

.depend: $(SRCS)
	$(CXX) $(CXXFLAGS) -MM $(SRCS) > .depend;

clean-fpga:
	rm -rf $(PROJECT) *.o .depend

clean-asesim:
	rm -rf $(PROJECT_ASE) *.o .depend

clean-vlsim:
	$(MAKE) -C vlsim clean

clean: clean-fpga clean-asesim clean-vlsim

ifneq ($(MAKECMDGOALS),clean)
    -include .depend
endif
