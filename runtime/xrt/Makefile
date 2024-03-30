include ../common.mk

SRC_DIR := $(VORTEX_HOME)/runtime/xrt

CXXFLAGS += -std=c++14 -Wall -Wextra -Wfatal-errors
CXXFLAGS += -I$(INC_DIR) -I$(COMMON_DIR) -I$(ROOT_DIR)/hw -I$(XILINX_XRT)/include -I$(SIM_DIR)/common
CXXFLAGS += -fPIC

LDFLAGS += -shared -pthread
LDFLAGS += -L$(XILINX_XRT)/lib -luuid -lxrt_coreutil

SRCS := $(SRC_DIR)/vortex.cpp $(COMMON_DIR)/utils.cpp $(SIM_DIR)/common/util.cpp

PROJECT := libvortex.so

# Debugigng
ifdef DEBUG
	CXXFLAGS += -g -O0
else    
	CXXFLAGS += -O2 -DNDEBUG
endif

# Enable scope logic analyzer
ifdef SCOPE
	CXXFLAGS += -DSCOPE	
	SRCS += $(COMMON_DIR)/scope.cpp
endif

all: $(PROJECT)

$(PROJECT): $(SRCS) $(SCOPE_JSON)
	$(CXX) $(CXXFLAGS) $^ $(LDFLAGS) -o $@

clean:
	rm -rf $(PROJECT) obj_dir
