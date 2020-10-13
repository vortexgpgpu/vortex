OPAE_HOME ?= /tools/opae/1.4.0

#CXXFLAGS += -std=c++11 -O2 -DNDEBUG -Wall -Wextra -pedantic -Wfatal-errors
CXXFLAGS += -std=c++11 -O0 -g -Wall -Wextra -pedantic -Wfatal-errors

CXXFLAGS += -I../include -I$(OPAE_HOME)/include  -I../../hw

LDFLAGS += -L$(OPAE_HOME)/lib

SCOPE=1

# stack execution protection
LDFLAGS +=-z noexecstack

# data relocation and projection
LDFLAGS +=-z relro -z now

# stack buffer overrun detection
CXXFLAGS +=-fstack-protector

# Position independent code
CXXFLAGS += -fPIC

# Dump perf stats
CXXFLAGS += -DDUMP_PERF_STATS

LDFLAGS += -shared

FPGA_LIBS += -luuid -lopae-c

ASE_LIBS += -luuid -lopae-c-ase

VLSIM_LIBS += -lopae-c-vlsim

LIB_DIR=../lib

ASE_DIR = ase

VLSIM_DIR = vlsim

PROJECT = libvortex.so

PROJECT_ASE = $(ASE_DIR)/libvortex.so

PROJECT_VLSIM = $(VLSIM_DIR)/libvortex.so

AFU_JSON_INFO = vortex_afu.h

SRCS = vortex.cpp ../common/vx_utils.cpp

# Enable scope analyzer
ifdef SCOPE
	CXXFLAGS += -DSCOPE	
	SRCS += vx_scope.cpp
	SET_SCOPE = SCOPE=1
endif

all: vlsim

# AFU info from JSON file, including AFU UUID
json: ../../hw/opae/vortex_afu.json
	afu_json_mgr json-info --afu-json=$^ --c-hdr=$@

fpga: $(SRCS)
	$(CXX) $(CXXFLAGS) $^ $(LDFLAGS) $(FPGA_LIBS) -o $(PROJECT)

asesim: $(SRCS) $(ASE_DIR)
	$(CXX) $(CXXFLAGS) -DUSE_ASE $(SRCS) $(LDFLAGS) $(ASE_LIBS) -o $(PROJECT_ASE)

vlsim: $(SRCS) opae-vlsim
	$(CXX) $(CXXFLAGS) -L./vlsim -DUSE_VLSIM $(SRCS) $(LDFLAGS) $(VLSIM_LIBS) -o $(PROJECT_VLSIM)

opae-vlsim:
	$(SET_SCOPE) $(MAKE) -C vlsim

vortex.o: vortex.cpp
	$(CXX) $(CXXFLAGS) -c vortex.cpp -o $@

$(ASE_DIR):
	mkdir -p ase

.depend: $(SRCS)
	$(CXX) $(CXXFLAGS) -MM $(SRCS) > .depend;

clean:
	rm -rf $(PROJECT) $(PROJECT_ASE) $(PROJECT_VLSIM) *.o .depend
	$(MAKE) -C vlsim clean	

ifneq ($(MAKECMDGOALS),clean)
    -include .depend
endif
