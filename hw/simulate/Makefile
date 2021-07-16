CFLAGS += -std=c++11 -O2 -Wall -Wextra -Wfatal-errors
#CFLAGS += -std=c++11 -g -O0 -Wall -Wextra -Wfatal-errors

CFLAGS += -Wno-maybe-uninitialized

CFLAGS += -I../..

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

SINGLECORE = -DNUM_CLUSTERS=1 -DNUM_CORES=1 -DL2_ENABLE=0

#MULTICORE = -DNUM_CLUSTERS=2 -DNUM_CORES=4 -DL2_ENABLE=1
#MULTICORE = -DNUM_CLUSTERS=1 -DNUM_CORES=4 -DL2_ENABLE=1
MULTICORE = -DNUM_CLUSTERS=1 -DNUM_CORES=2 -DL2_ENABLE=0

SINGLECORE += $(CONFIGS)
MULTICORE += $(CONFIGS)

TOP = Vortex

RTL_DIR=../rtl
DPI_DIR=../dpi

FPU_INCLUDE = -I$(RTL_DIR)/fp_cores -I$(DPI_DIR) -I$(RTL_DIR)/fp_cores/fpnew/src/common_cells/include -I$(RTL_DIR)/fp_cores/fpnew/src/common_cells/src -I$(RTL_DIR)/fp_cores/fpnew/src/fpu_div_sqrt_mvp/hdl -I$(RTL_DIR)/fp_cores/fpnew/src 
RTL_INCLUDE = -I$(RTL_DIR)/ -I$(RTL_DIR)/libs -I$(RTL_DIR)/interfaces -I$(RTL_DIR)/cache -I$(RTL_DIR)/simulate $(FPU_INCLUDE)

SRCS = simulator.cpp main.cpp
SRCS += $(DPI_DIR)/util_dpi.cpp $(DPI_DIR)/float_dpi.cpp

VL_FLAGS += -O2 --language 1800-2009 --assert -Wall -Wpedantic
VL_FLAGS += -Wno-DECLFILENAME -Wno-REDEFMACRO
VL_FLAGS += --x-initial unique --x-assign unique
VL_FLAGS += verilator.vlt

VL_FLAGS += --exe $(SRCS) $(RTL_INCLUDE)
VL_FLAGS += --cc Vortex.v --top-module $(TOP)

# FPU backend
FPU_CORE ?= FPU_FPNEW
VL_FLAGS += -D$(FPU_CORE)

DBG_FLAGS += -DVCD_OUTPUT

THREADS ?= $(shell python3 -c 'import multiprocessing as mp; print(max(1, mp.cpu_count() // 2))')

all: build-s

gen-s:
	verilator $(VL_FLAGS) -DNDEBUG  $(SINGLECORE) -CFLAGS '$(CFLAGS) -DNDEBUG $(SINGLECORE)'

gen-sd:
	verilator $(VL_FLAGS) $(SINGLECORE) -CFLAGS '$(CFLAGS) $(DBG_FLAGS) $(SINGLECORE)' --trace --trace-structs $(DBG_FLAGS)

gen-st:
	verilator $(VL_FLAGS) -DNDEBUG $(SINGLECORE) -CFLAGS '$(CFLAGS) -DNDEBUG $(SINGLECORE)' --threads $(THREADS)

gen-m:
	verilator $(VL_FLAGS) -DNDEBUG $(MULTICORE) -CFLAGS '$(CFLAGS) -DNDEBUG $(MULTICORE)'
	
gen-md:
	verilator $(VL_FLAGS) $(MULTICORE) -CFLAGS '$(CFLAGS) $(DBG_FLAGS) $(MULTICORE)' --trace --trace-structs $(DBG_FLAGS)

gen-mt:
	verilator $(VL_FLAGS) -DNDEBUG $(MULTICORE) -CFLAGS '$(CFLAGS) -DNDEBUG $(MULTICORE)' --threads $(THREADS)

build-s: gen-s
	make -j -C obj_dir -f VVortex.mk

build-sd: gen-sd
	make -j -C obj_dir -f VVortex.mk

build-st: gen-st
	make -j -C obj_dir -f VVortex.mk

build-m: gen-m
	make -j -C obj_dir -f VVortex.mk

build-md: gen-md
	make -j -C obj_dir -f VVortex.mk

build-mt: gen-mt
	make -j -C obj_dir -f VVortex.mk

clean:
	rm -rf obj_dir
