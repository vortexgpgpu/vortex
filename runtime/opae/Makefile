TARGET ?= opaesim

OPAESIM_DIR = ../../sim/opaesim

RTL_DIR=../../hw/rtl

SYN_DIR=../../hw/syn/altera/opae

SCRIPT_DIR=../../hw/scripts

CXXFLAGS += -std=c++11 -Wall -Wextra -pedantic -Wfatal-errors

CXXFLAGS += -I. -I../include -I../common/ -I../../hw

ifeq ($(TARGET), opaesim)
	CXXFLAGS += -I$(OPAESIM_DIR)
else
	CXXFLAGS += -I$(SYN_DIR)
endif

# Position independent code
CXXFLAGS += -fPIC

# Add external configuration
CXXFLAGS += $(CONFIGS)

# Dump perf stats
CXXFLAGS += -DDUMP_PERF_STATS

LDFLAGS += -shared -luuid -ldl -pthread

SRCS = vortex.cpp driver.cpp ../common/utils.cpp

# set up target types
ifeq ($(TARGET), opaesim)
	CXXFLAGS += -DOPAESIM
else
	ifeq ($(TARGET), asesim)
		CXXFLAGS += -DASESIM
	else
		CXXFLAGS += -DFPGA
	endif
endif

# Debugigng
ifdef DEBUG
	CXXFLAGS += -g -O0
else    
	CXXFLAGS += -O2 -DNDEBUG
	SCOPE_FLAGS += -D NDEBUG
endif

# Enable scope analyzer
ifdef SCOPE
	CXXFLAGS += -DSCOPE	
	SRCS += ../common/scope.cpp
	SCOPE_H = scope-defs.h	 
	SCOPE_FLAGS += -D SCOPE
endif

# Enable perf counters
ifdef PERF
	CXXFLAGS += -DPERF_ENABLE
endif

PROJECT = libvortex.so

all: $(PROJECT)

scope-defs.h: $(SCRIPT_DIR)/scope.json
	$(SCRIPT_DIR)/scope.py $(CONFIGS) $(SCOPE_FLAGS) -cc scope-defs.h -vl $(RTL_DIR)/scope-defs.vh $(SCRIPT_DIR)/scope.json

# generate scope data
scope: scope-defs.h

$(PROJECT): $(SRCS) $(SCOPE_H)
ifeq ($(TARGET), opaesim)
	$(MAKE) -C $(OPAESIM_DIR)
endif
	$(CXX) $(CXXFLAGS) $(SRCS) $(LDFLAGS) -o $(PROJECT)

clean:
	rm -rf $(PROJECT) *.o scope-defs.h $(RTL_DIR)/scope-defs.vh
ifeq ($(TARGET), opaesim)
	$(MAKE) -C $(OPAESIM_DIR) clean
endif
