CXXFLAGS += -std=c++14 -Wall -Wextra -Wfatal-errors

CXXFLAGS += -I../include -I../common -I../../hw -I$(XILINX_XRT)/include -I../../sim/common

CXXFLAGS += -fPIC

LDFLAGS += -shared -pthread
LDFLAGS += -L$(XILINX_XRT)/lib -luuid -lxrt_coreutil

SRCS = vortex.cpp ../common/utils.cpp ../../sim/common/util.cpp

PROJECT = libvortex.so

# Debugigng
ifdef DEBUG
	CXXFLAGS += -g -O0
else    
	CXXFLAGS += -O2 -DNDEBUG
endif

# Enable scope logic analyzer
ifdef SCOPE
	CXXFLAGS += -DSCOPE	
	SRCS += ../common/scope.cpp
endif

all: $(PROJECT)

$(PROJECT): $(SRCS) $(SCOPE_JSON)
	$(CXX) $(CXXFLAGS) $^ $(LDFLAGS) -o $@

clean:
	rm -rf $(PROJECT) obj_dir
