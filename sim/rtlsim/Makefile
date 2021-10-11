RTL_DIR=../../hw/rtl
DPI_DIR=../../hw/dpi

CXXFLAGS += -std=c++11 -Wall -Wextra -Wfatal-errors
CXXFLAGS += -fPIC -Wno-maybe-uninitialized
CXXFLAGS += -I../../../hw -I../../common
CXXFLAGS += -I../../common/softfloat/source/include

LDFLAGS += ../../common/softfloat/build/Linux-x86_64-GCC/softfloat.a

# control RTL debug print states
DBG_PRINT_FLAGS += -DDBG_PRINT_PIPELINE  
DBG_PRINT_FLAGS += -DDBG_PRINT_CORE_ICACHE
DBG_PRINT_FLAGS += -DDBG_PRINT_CORE_DCACHE
DBG_PRINT_FLAGS += -DDBG_PRINT_CACHE_BANK 
DBG_PRINT_FLAGS += -DDBG_PRINT_CACHE_MSHR
DBG_PRINT_FLAGS += -DDBG_PRINT_CACHE_TAG
DBG_PRINT_FLAGS += -DDBG_PRINT_CACHE_DATA
DBG_PRINT_FLAGS += -DDBG_PRINT_MEM
DBG_PRINT_FLAGS += -DDBG_PRINT_OPAE
DBG_PRINT_FLAGS += -DDBG_PRINT_AVS
DBG_PRINT_FLAGS += -DDBG_PRINT_SCOPE

DBG_FLAGS += $(DBG_PRINT_FLAGS)
DBG_FLAGS += -DDBG_CACHE_REQ_INFO
DBG_FLAGS += -DVCD_OUTPUT

FPU_INCLUDE = -I$(RTL_DIR)/fp_cores -I$(RTL_DIR)/fp_cores/fpnew/src/common_cells/include -I$(RTL_DIR)/fp_cores/fpnew/src/common_cells/src -I$(RTL_DIR)/fp_cores/fpnew/src/fpu_div_sqrt_mvp/hdl -I$(RTL_DIR)/fp_cores/fpnew/src
RTL_INCLUDE = -I$(RTL_DIR) -I$(DPI_DIR) -I$(RTL_DIR)/libs -I$(RTL_DIR)/interfaces -I$(RTL_DIR)/cache -I$(RTL_DIR)/simulate $(FPU_INCLUDE)

SRCS = ../common/util.cpp ../common/mem.cpp ../common/rvfloats.cpp
SRCS += $(DPI_DIR)/util_dpi.cpp $(DPI_DIR)/float_dpi.cpp
SRCS += main.cpp simulator.cpp

ifdef AXI_BUS
	TOP = Vortex_axi
	CFLAGS += -DAXI_BUS
else
	TOP = Vortex
endif

VL_FLAGS = --exe --cc $(TOP) --top-module $(TOP)
VL_FLAGS += -O2 --language 1800-2009 --assert -Wall -Wpedantic
VL_FLAGS += -Wno-DECLFILENAME -Wno-REDEFMACRO
VL_FLAGS += --x-initial unique --x-assign unique
VL_FLAGS += verilator.vlt
VL_FLAGS += $(CONFIGS)
VL_FLAGS += $(RTL_INCLUDE)

# Debugigng
ifdef DEBUG
	VL_FLAGS += -DVCD_OUTPUT --trace --trace-structs $(DBG_FLAGS)
	CXXFLAGS += -g -O0 -DVCD_OUTPUT $(DBG_FLAGS)
else    
	VL_FLAGS += -DNDEBUG
	CXXFLAGS += -O2 -DNDEBUG
endif

# Enable perf counters
ifdef PERF
	VL_FLAGS += -DPERF_ENABLE
	CXXFLAGS += -DPERF_ENABLE
endif

# ALU backend
VL_FLAGS += -DIMUL_DPI
VL_FLAGS += -DIDIV_DPI

# FPU backend
FPU_CORE ?= FPU_DPI
VL_FLAGS += -D$(FPU_CORE)

THREADS ?= $(shell python3 -c 'import multiprocessing as mp; print(max(1, mp.cpu_count() // 2))')

PROJECT = rtlsim

all: $(PROJECT)

$(PROJECT): $(SRCS)
	verilator --build $(VL_FLAGS) $(SRCS) -CFLAGS '$(CXXFLAGS)' -LDFLAGS '$(LDFLAGS)' -o ../$(PROJECT)
	
static: $(SRCS)
	verilator --build $(VL_FLAGS) $(SRCS) -CFLAGS '$(CXXFLAGS)' -LDFLAGS '$(LDFLAGS)'
	$(AR) rcs lib$(PROJECT).a obj_dir/*.o ../common/softfloat/build/Linux-x86_64-GCC/*.o

clean-static:
	rm -rf lib$(PROJECT).a obj_dir

clean: clean-static
	rm -rf $(PROJECT)
