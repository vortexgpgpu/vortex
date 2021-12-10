DESTDIR ?= .
RTL_DIR = ../../hw/rtl
DPI_DIR = ../../hw/dpi
SCRIPT_DIR = ../../hw/scripts
THIRD_PARTY_DIR = ../../third_party

CXXFLAGS += -std=c++11 -Wall -Wextra -Wfatal-errors -Wno-array-bounds
CXXFLAGS += -fPIC -Wno-maybe-uninitialized
CXXFLAGS += -I.. -I../../../hw -I../../common
CXXFLAGS += -I../$(THIRD_PARTY_DIR)/softfloat/source/include
CXXFLAGS += -I../$(THIRD_PARTY_DIR)

LDFLAGS += -shared ../$(THIRD_PARTY_DIR)/softfloat/build/Linux-x86_64-GCC/softfloat.a
LDFLAGS += -L../$(THIRD_PARTY_DIR)/ramulator -lramulator

# control RTL debug tracing states
DBG_TRACE_FLAGS += -DDBG_TRACE_PIPELINE  
DBG_TRACE_FLAGS += -DDBG_TRACE_CORE_ICACHE
DBG_TRACE_FLAGS += -DDBG_TRACE_CORE_DCACHE
DBG_TRACE_FLAGS += -DDBG_TRACE_CACHE_BANK 
DBG_TRACE_FLAGS += -DDBG_TRACE_CACHE_MSHR
DBG_TRACE_FLAGS += -DDBG_TRACE_CACHE_TAG
DBG_TRACE_FLAGS += -DDBG_TRACE_CACHE_DATA
DBG_TRACE_FLAGS += -DDBG_TRACE_MEM
DBG_TRACE_FLAGS += -DDBG_TRACE_OPAE
DBG_TRACE_FLAGS += -DDBG_TRACE_AVS
DBG_TRACE_FLAGS += -DDBG_TRACE_SCOPE
DBG_TRACE_FLAGS += -DDBG_TRACE_TEX

DBG_FLAGS += $(DBG_TRACE_FLAGS)

SRCS = ../common/util.cpp ../common/mem.cpp ../common/rvfloats.cpp
SRCS += $(DPI_DIR)/util_dpi.cpp $(DPI_DIR)/float_dpi.cpp
SRCS += fpga.cpp opae_sim.cpp

FPU_INCLUDE = -I$(RTL_DIR)/fp_cores -I$(THIRD_PARTY_DIR)/fpnew/src/common_cells/include -I$(THIRD_PARTY_DIR)/fpnew/src/common_cells/src -I$(THIRD_PARTY_DIR)/fpnew/src/fpu_div_sqrt_mvp/hdl -I$(THIRD_PARTY_DIR)/fpnew/src
TEX_INCLUDE = -I$(RTL_DIR)/tex_unit
RTL_INCLUDE = -I$(RTL_DIR) -I$(DPI_DIR) -I$(RTL_DIR)/libs -I$(RTL_DIR)/interfaces -I$(RTL_DIR)/cache $(FPU_INCLUDE) $(TEX_INCLUDE)
RTL_INCLUDE += -I$(RTL_DIR)/afu -I$(RTL_DIR)/afu/ccip

TOP = vortex_afu_shim

VL_FLAGS = --exe --cc $(TOP) --top-module $(TOP)
VL_FLAGS += -O2 --language 1800-2009 --assert -Wall -Wpedantic
VL_FLAGS += -Wno-DECLFILENAME -Wno-REDEFMACRO
VL_FLAGS += --x-initial unique --x-assign unique
VL_FLAGS += verilator.vlt
VL_FLAGS += $(RTL_INCLUDE)

VL_FLAGS += $(CONFIGS)
CXXFLAGS += $(CONFIGS)

# Enable Verilator multithreaded simulation
#THREADS ?= $(shell python3 -c 'import multiprocessing as mp; print(max(1, mp.cpu_count() // 2))')
#VL_FLAGS += --threads $(THREADS)

# Enable VCD trace
#VCD_TRACE = -DVCD_OUTPUT

# Debugigng
ifdef DEBUG
	VL_FLAGS += $(VCD_TRACE) --trace --trace-structs $(DBG_FLAGS)
	CXXFLAGS += -g -O0 $(VCD_TRACE) $(DBG_FLAGS)
else    
	VL_FLAGS += -DNDEBUG
	CXXFLAGS += -O2 -DNDEBUG
endif

# Enable scope analyzer
ifdef SCOPE
	VL_FLAGS += -DSCOPE
	CXXFLAGS += -DSCOPE
endif

# Enable perf counters
ifdef PERF
	VL_FLAGS += -DPERF_ENABLE
	CXXFLAGS += -DPERF_ENABLE
endif

# use our OPAE shim
VL_FLAGS += -DNOPAE
CXXFLAGS += -DNOPAE

# ALU backend
VL_FLAGS += -DIMUL_DPI
VL_FLAGS += -DIDIV_DPI

# FPU backend
FPU_CORE ?= FPU_DPI
VL_FLAGS += -D$(FPU_CORE)

PROJECT = libopae-c-vlsim.so

all: $(PROJECT)

vortex_afu.h : $(RTL_DIR)/afu/vortex_afu.vh
	$(SCRIPT_DIR)/gen_config.py -i $(RTL_DIR)/afu/vortex_afu.vh -o vortex_afu.h

$(DESTDIR)/$(PROJECT): $(SRCS) vortex_afu.h
	verilator --build $(VL_FLAGS) $(SRCS) -CFLAGS '$(CXXFLAGS)' -LDFLAGS '$(LDFLAGS)' -o ../$(DESTDIR)/$(PROJECT)

clean:
	rm -rf obj_dir $(DESTDIR)/$(PROJECT)
