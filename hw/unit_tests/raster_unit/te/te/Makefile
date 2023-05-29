PARAMS ?=
DEBUG ?= 1

RTL_DIR = ../../../../rtl
DPI_DIR = ../../../../dpi

# control RTL debug tracing states
DBG_TRACE_FLAGS += -DDBG_TRACE_CORE_PIPELINE  
DBG_TRACE_FLAGS += -DDBG_TRACE_CORE_ICACHE
DBG_TRACE_FLAGS += -DDBG_TRACE_CORE_DCACHE
DBG_TRACE_FLAGS += -DDBG_TRACE_CORE_MEM
DBG_TRACE_FLAGS += -DDBG_TRACE_CACHE_BANK 
DBG_TRACE_FLAGS += -DDBG_TRACE_CACHE_MSHR
DBG_TRACE_FLAGS += -DDBG_TRACE_CACHE_TAG
DBG_TRACE_FLAGS += -DDBG_TRACE_CACHE_DATA
DBG_TRACE_FLAGS += -DDBG_TRACE_AFU
DBG_TRACE_FLAGS += -DDBG_TRACE_SCOPE
DBG_TRACE_FLAGS += -DDBG_TRACE_TEX
DBG_TRACE_FLAGS += -DDBG_TRACE_RASTER
DBG_TRACE_FLAGS += -DDBG_TRACE_ROP

DBG_FLAGS += $(DBG_TRACE_FLAGS)

FPU_INCLUDE = -I$(RTL_DIR)/fpu -I$(THIRD_PARTY_DIR)/fpnew/src/common_cells/include -I$(THIRD_PARTY_DIR)/fpnew/src/common_cells/src -I$(THIRD_PARTY_DIR)/fpnew/src/fpu_div_sqrt_mvp/hdl -I$(THIRD_PARTY_DIR)/fpnew/src
TEX_INCLUDE = -I$(RTL_DIR)/tex
RASTER_INCLUDE = -I$(RTL_DIR)/raster -I$(RTL_DIR)/raster/tile_evaluator
ROP_INCLUDE = -I$(RTL_DIR)/rop
RTL_INCLUDE = -I$(RTL_DIR) -I$(DPI_DIR) -I$(RTL_DIR)/libs -I$(RTL_DIR)/interfaces -I$(RTL_DIR)/cache -I$(RTL_DIR)/simulate $(FPU_INCLUDE) -I$(DPI_DIR)/..
RTL_INCLUDE += $(TEX_INCLUDE) $(RASTER_INCLUDE) $(ROP_INCLUDE)

TOP = VX_raster_te
TESTBENCH = testbench.cpp
SRCS = $(TESTBENCH) $(DPI_DIR)/util_dpi.cpp

CXXFLAGS += -std=c++11 -Wall -Wextra -Wfatal-errors -Wno-array-bounds -Wno-maybe-uninitialized
CXXFLAGS += -I$(DPI_DIR)/.. -I$(DPI_DIR) -I$(DPI_DIR)/../common -I/nethome/vsaxena36/vortex-dev/hw

VL_FLAGS += --exe --cc $(TOP).sv  --top-module $(TOP)
VL_FLAGS += --language 1800-2009 --assert -Wall
VL_FLAGS += -Wno-DECLFILENAME -Wno-REDEFMACRO
VL_FLAGS += --x-initial unique --x-assign unique -Wno-UNUSED -Wno-UNDRIVEN
VL_FLAGS += $(RTL_INCLUDE)
VL_FLAGS += $(PARAMS)

# Enable Verilator multithreaded simulation
THREADS ?= $(shell python -c 'import multiprocessing as mp; print(mp.cpu_count())')
VL_FLAGS += -j $(THREADS)
#VL_FLAGS += --threads $(THREADS)

# Debugigng
ifdef DEBUG
	VL_FLAGS += --trace --trace-structs -DVCD_OUTPUT $(DBG_FLAGS)
	CXXFLAGS += -g -O0 -DVCD_OUTPUT $(DBG_FLAGS)
else    
	VL_FLAGS += -DNDEBUG
	CXXFLAGS += -O2 -DNDEBUG
endif

all: $(TOP)

$(TOP): $(SRCS)
	verilator --build $(VL_FLAGS) $^ -CFLAGS '$(CXXFLAGS)' -o ../$(TOP)
	
run: $(TOP)
	./$(TOP)

clean:
	rm -rf obj_dir $(TOP)