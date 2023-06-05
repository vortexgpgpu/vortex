PARAMS ?=
DEBUG ?= 1

RTL_DIR = ../../../rtl
DPI_DIR = ../../../dpi

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

DBG_FLAGS += $(DBG_TRACE_FLAGS) -DVCD_OUTPUT

FPU_INCLUDE = -I$(RTL_DIR)/fpu -I$(THIRD_PARTY_DIR)/fpnew/src/common_cells/include -I$(THIRD_PARTY_DIR)/fpnew/src/common_cells/src -I$(THIRD_PARTY_DIR)/fpnew/src/fpu_div_sqrt_mvp/hdl -I$(THIRD_PARTY_DIR)/fpnew/src
TEX_INCLUDE = -I$(RTL_DIR)/tex
RASTER_INCLUDE = -I$(RTL_DIR)/raster
ROP_INCLUDE = -I$(RTL_DIR)/rop
RTL_INCLUDE = -I$(RTL_DIR) -I$(DPI_DIR) -I$(RTL_DIR)/libs -I$(RTL_DIR)/interfaces -I$(RTL_DIR)/cache -I$(RTL_DIR)/simulate $(FPU_INCLUDE)
RTL_INCLUDE += $(TEX_INCLUDE) $(RASTER_INCLUDE) $(ROP_INCLUDE)

TOP = VX_tex_sampler

SRCS = main.cpp ../../../dpi/util_dpi.cpp

CXXFLAGS += -std=c++11 -Wall -Wextra -Wfatal-errors -Wno-array-bounds -Wno-maybe-uninitialized
CXXFLAGS += -I../../../.. -I../../../../dpi -I../../../common

VL_FLAGS += --exe --cc $(TOP).sv 
VL_FLAGS += --language 1800-2009 --assert -Wall
VL_FLAGS += -Wno-DECLFILENAME -Wno-REDEFMACRO
VL_FLAGS += --x-initial unique --x-assign unique
VL_FLAGS += $(RTL_INCLUDE)
VL_FLAGS += $(PARAMS)

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

all: $(TOP)

$(TOP): $(SRCS)
	verilator --build $(VL_FLAGS) $^ -CFLAGS '$(CXXFLAGS)' -o ../$(TOP)
	
run: $(TOP)
	./$(TOP)

clean:
	rm -rf obj_dir $(TOP)
