CXXFLAGS += -std=c++11 -Wall -Wextra -Wfatal-errors

CXXFLAGS += -I../include -I../common -I../../hw -I$(XILINX_XRT)/include -I../../sim/common

CXXFLAGS += -fPIC

LDFLAGS += -shared -pthread
LDFLAGS += -L$(XILINX_XRT)/lib -luuid -lxrt_coreutil

SRCS = vortex.cpp ../common/vx_utils.cpp ../../sim/common/util.cpp

PROJECT = libvortex.so

# Debugigng
ifdef DEBUG
	CXXFLAGS += -g -O0
else    
	CXXFLAGS += -O2 -DNDEBUG
endif

all: $(PROJECT)

$(PROJECT): $(SRCS) 
	$(CXX) $(CXXFLAGS) $^ $(LDFLAGS) -o $@

clean:
	rm -rf $(PROJECT) obj_dir