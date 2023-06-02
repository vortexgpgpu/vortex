XLEN ?= 32

TARGET ?= opaesim

OPAESIM_DIR = ../../sim/opaesim

RTL_DIR=../../hw/rtl

SYN_DIR=../../hw/syn/altera/opae

SCRIPT_DIR=../../hw/scripts

CXXFLAGS += -std=c++11 -Wall -Wextra -pedantic -Wfatal-errors
CXXFLAGS += -I. -I../include -I../common/ -I../../hw
CXXFLAGS += -DXLEN_$(XLEN)

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
	OPAESIM = libopae-c-sim.so
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
endif

# Enable scope logic analyzer
ifdef SCOPE
	CXXFLAGS += -DSCOPE	
	SRCS += ../common/scope.cpp
endif

# Enable perf counters
ifdef PERF
	CXXFLAGS += -DPERF_ENABLE
endif

PROJECT = libvortex.so

all: $(PROJECT)

libopae-c-sim.so:
	DESTDIR=../../runtime/opae $(MAKE) -C $(OPAESIM_DIR) ../../runtime/opae/libopae-c-sim.so

$(PROJECT): $(SRCS) $(OPAESIM)
	$(CXX) $(CXXFLAGS) $(SRCS) $(LDFLAGS) -o $(PROJECT)

clean:
	DESTDIR=../../runtime/opae $(MAKE) -C $(OPAESIM_DIR) clean
	rm -rf $(PROJECT)
