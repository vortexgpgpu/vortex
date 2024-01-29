DESTDIR ?= .
RTL_DIR = ../../rtl
DPI_DIR = ../../dpi

CONFIGS +=
PARAMS +=

CXXFLAGS += -std=c++11 -Wall -Wextra -Wfatal-errors -Wno-array-bounds
CXXFLAGS += -fPIC -Wno-maybe-uninitialized
CXXFLAGS += -I../../.. -I../../common -I../../../../sim/common
CXXFLAGS += $(CONFIGS)

LDFLAGS += 

DBG_FLAGS += -DDEBUG_LEVEL=$(DEBUG) -DVCD_OUTPUT $(DBG_TRACE_FLAGS)

RTL_PKGS = $(RTL_DIR)/VX_gpu_pkg.sv $(RTL_DIR)/fpu/VX_fpu_pkg.sv

RTL_INCLUDE = -I$(RTL_DIR) -I$(DPI_DIR) -I$(RTL_DIR)/libs -I$(RTL_DIR)/interfaces -I$(RTL_DIR)/mem -I$(RTL_DIR)/cache -I$(RTL_DIR)/fpu -I$(RTL_DIR)/core

SRCS = main.cpp
SRCS += $(DPI_DIR)/util_dpi.cpp

VL_FLAGS = --exe
VL_FLAGS += --language 1800-2009 --assert -Wall -Wpedantic
VL_FLAGS += -Wno-DECLFILENAME -Wno-REDEFMACRO
VL_FLAGS += --x-initial unique --x-assign unique
VL_FLAGS += -DSIMULATION -DSV_DPI
VL_FLAGS += $(CONFIGS)
VL_FLAGS += $(PARAMS)
VL_FLAGS += $(RTL_INCLUDE)
VL_FLAGS += $(RTL_PKGS)

# Enable Verilator multithreaded simulation
THREADS ?= $(shell python -c 'import multiprocessing as mp; print(mp.cpu_count())')
VL_FLAGS += -j $(THREADS)
#VL_FLAGS += --threads $(THREADS)

# Debugigng
ifdef DEBUG
	VL_FLAGS += --trace --trace-structs $(DBG_FLAGS)
	CXXFLAGS += -g -O0 $(DBG_FLAGS)
else    
	VL_FLAGS += -DNDEBUG
	CXXFLAGS += -O2 -DNDEBUG
endif

# Enable perf counters
ifdef PERF
	VL_FLAGS += -DPERF_ENABLE
	CXXFLAGS += -DPERF_ENABLE
endif

PROJECT = top_modules

all: build
	
build: $(SRCS)
	verilator --build $(VL_FLAGS) --cc VX_cache_top --top-module VX_cache_top $^ -CFLAGS '$(CXXFLAGS)'
	verilator --build $(VL_FLAGS) --cc VX_core_top --top-module VX_core_top $^ -CFLAGS '$(CXXFLAGS)'

run: 

waves: 

clean:
	rm -rf obj_dir
