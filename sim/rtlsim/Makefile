CXXFLAGS += -std=c++11 -O2 -Wall -Wextra -Wfatal-errors
#CXXFLAGS += -std=c++11 -g -O0 -Wall -Wextra -Wfatal-errors

CXXFLAGS += -fPIC -Wno-maybe-uninitialized
CXXFLAGS += -I. -I../../../hw -I../../common
CXXFLAGS += -I$(VERILATOR_ROOT)/include -I$(VERILATOR_ROOT)/include/vltstd

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

SINGLECORE = -DNUM_CLUSTERS=1 -DNUM_CORES=1 -DL2_ENABLE=0
MULTICORE  = -DNUM_CLUSTERS=1 -DNUM_CORES=2 -DL2_ENABLE=0

RTL_DIR=../../hw/rtl
DPI_DIR=../../hw/dpi

FPU_INCLUDE = -I$(RTL_DIR)/fp_cores -I$(RTL_DIR)/fp_cores/fpnew/src/common_cells/include -I$(RTL_DIR)/fp_cores/fpnew/src/common_cells/src -I$(RTL_DIR)/fp_cores/fpnew/src/fpu_div_sqrt_mvp/hdl -I$(RTL_DIR)/fp_cores/fpnew/src
RTL_INCLUDE = -I$(RTL_DIR) -I$(DPI_DIR) -I$(RTL_DIR)/libs -I$(RTL_DIR)/interfaces -I$(RTL_DIR)/cache -I$(RTL_DIR)/simulate $(FPU_INCLUDE)

SRCS = ../common/util.cpp ../common/mem.cpp
SRCS += $(DPI_DIR)/util_dpi.cpp $(DPI_DIR)/float_dpi.cpp
SRCS += simulator.cpp

ifdef AXI_BUS
	TOP = Vortex_axi
	CFLAGS += -DAXI_BUS
else
	TOP = Vortex
endif

VL_FLAGS = --cc $(TOP) --top-module $(TOP)
VL_FLAGS += -O2 --language 1800-2009 --assert -Wall -Wpedantic
VL_FLAGS += -Wno-DECLFILENAME -Wno-REDEFMACRO
VL_FLAGS += --x-initial unique --x-assign unique
VL_FLAGS += verilator.vlt
VL_FLAGS += $(CONFIGS)
VL_FLAGS += $(RTL_INCLUDE)

# Debugigng
ifdef DEBUG
	VL_FLAGS += -DVCD_OUTPUT --trace --trace-structs $(DBG_FLAGS)
	CXXFLAGS += -DVCD_OUTPUT $(DBG_FLAGS)
else    
	VL_FLAGS += -DNDEBUG
	CXXFLAGS += -DNDEBUG
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
FPU_CORE ?= FPU_FPNEW
VL_FLAGS += -D$(FPU_CORE)

THREADS ?= $(shell python3 -c 'import multiprocessing as mp; print(max(1, mp.cpu_count() // 2))')

OBJS := $(patsubst %.cpp, obj_dir/%.o, $(notdir $(SRCS)))
VPATH := $(sort $(dir $(SRCS)))

#$(info OBJS is $(OBJS))
#$(info VPATH is $(VPATH))

PROJECT = rtlsim

all: build-s

build-s:
	verilator --build --exe main.cpp $(SRCS) $(VL_FLAGS) -DNDEBUG  $(SINGLECORE) -CFLAGS '$(CXXFLAGS) -DNDEBUG $(SINGLECORE)' -o ../$(PROJECT)
	
build-sd: 
	verilator --build --exe main.cpp $(SRCS) $(VL_FLAGS) $(SINGLECORE) -CFLAGS '$(CXXFLAGS) $(DBG_FLAGS) $(SINGLECORE)' --trace --trace-structs $(DBG_FLAGS) -o ../$(PROJECT)

build-st:
	verilator --build --exe main.cpp $(SRCS) $(VL_FLAGS) -DNDEBUG $(SINGLECORE) -CFLAGS '$(CXXFLAGS) -DNDEBUG $(SINGLECORE)' --threads $(THREADS) -o ../$(PROJECT)

build-m:
	verilator --build --exe main.cpp $(SRCS) $(VL_FLAGS) -DNDEBUG $(MULTICORE) -CFLAGS '$(CXXFLAGS) -DNDEBUG $(MULTICORE)' -o ../$(PROJECT)

build-md:
	verilator --build --exe main.cpp $(SRCS) $(VL_FLAGS) $(MULTICORE) -CFLAGS '$(CXXFLAGS) $(DBG_FLAGS) $(MULTICORE)' --trace --trace-structs $(DBG_FLAGS) -o ../$(PROJECT)

build-mt:
	verilator --build --exe main.cpp $(SRCS) $(VL_FLAGS) -DNDEBUG $(MULTICORE) -CFLAGS '$(CXXFLAGS) -DNDEBUG $(MULTICORE)' --threads $(THREADS) -o ../$(PROJECT)

obj_dir/V$(TOP)__ALL.a:
	verilator --build $(VL_FLAGS) -CFLAGS '$(CXXFLAGS)'

obj_dir/%.o: %.cpp
	cd obj_dir && $(CXX) $(CXXFLAGS) -c ../$< -o $(notdir $@)

obj_dir/verilated.o: $(VERILATOR_ROOT)/include/verilated.cpp
	cd obj_dir && $(CXX) $(CXXFLAGS) -c $< -o verilated.o

static: obj_dir/V$(TOP)__ALL.a $(OBJS) obj_dir/verilated.o
	cp obj_dir/V$(TOP)__ALL.a lib$(PROJECT).a
	$(AR) rs lib$(PROJECT).a $(OBJS) obj_dir/verilated.o

clean-objdir:
	rm -rf obj_dir

clean: clean-objdir
	rm -rf $(PROJECT) lib$(PROJECT).a
