
CXXFLAGS += -std=c++11 -O0 -g -Wall -Wextra -pedantic -Wfatal-errors

CXXFLAGS += -I../include -I/tools/opae/1.4.0/include  -I../../hw

LDFLAGS += -L/tools/opae/1.4.0/lib

# stack execution protection
LDFLAGS +=-z noexecstack

# data relocation and projection
LDFLAGS +=-z relro -z now

# stack buffer overrun detection
CXXFLAGS +=-fstack-protector

# Position independent code
CXXFLAGS += -fPIC

# Enable scope analyzer
CXXFLAGS += -DSCOPE

# config parameters
CXXFLAGS += -DNUM_WARPS=2 -DNUM_THREADS=2

LDFLAGS += -luuid

LDFLAGS += -shared

FPGA_LIBS += -lopae-c

ASE_LIBS += -lopae-c-ase

LIB_DIR=../lib

ASE_DIR = ase

PROJECT = libvortex.so

PROJECT_ASE = $(ASE_DIR)/libvortex.so

AFU_JSON_INFO = vortex_afu.h

SRCS = vortex.cpp scope.cpp ../common/vx_utils.cpp

all: $(PROJECT) $(PROJECT_ASE)

# AFU info from JSON file, including AFU UUID
$(AFU_JSON_INFO): ../../hw/opae/vortex_afu.json
	afu_json_mgr json-info --afu-json=$^ --c-hdr=$@

$(PROJECT): $(SRCS)
	$(CXX) $(CXXFLAGS) $^ $(LDFLAGS) $(FPGA_LIBS) -o $@

$(PROJECT_ASE): $(SRCS) $(ASE_DIR)
	$(CXX) $(CXXFLAGS) -DUSE_ASE $(SRCS) $(LDFLAGS) $(ASE_LIBS) -o $@

vortex.o: vortex.cpp $(AFU_JSON_INFO)
	$(CXX) $(CXXFLAGS) -c vortex.cpp -o $@

$(ASE_DIR):
	mkdir -p ase

.depend: $(SRCS) $(AFU_JSON_INFO)
	$(CXX) $(CXXFLAGS) -MM $(SRCS) > .depend;

clean:
	rm -rf $(PROJECT) $(PROJECT_ASE) $(AFU_JSON_INFO) *.o .depend 

ifneq ($(MAKECMDGOALS),clean)
    -include .depend
endif
