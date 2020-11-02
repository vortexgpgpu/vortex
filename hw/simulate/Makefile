SINGLECORE += -DNUM_CLUSTERS=1 -DNUM_CORES=1 

#MULTICORE += -DNUM_CLUSTERS=2 -DNUM_CORES=4 -DL2_ENABLE=1
#MULTICORE += -DNUM_CLUSTERS=1 -DNUM_CORES=4 -DL2_ENABLE=1
MULTICORE += -DNUM_CLUSTERS=1 -DNUM_CORES=2 -DL2_ENABLE=0

# control RTL debug print states
DBG_PRINT_FLAGS += -DDBG_PRINT_CORE_ICACHE
DBG_PRINT_FLAGS += -DDBG_PRINT_CORE_DCACHE
DBG_PRINT_FLAGS += -DDBG_PRINT_CACHE_BANK 
DBG_PRINT_FLAGS += -DDBG_PRINT_CACHE_SNP 
DBG_PRINT_FLAGS += -DDBG_PRINT_CACHE_MSRQ
DBG_PRINT_FLAGS += -DDBG_PRINT_DRAM      
DBG_PRINT_FLAGS += -DDBG_PRINT_PIPELINE  
DBG_PRINT_FLAGS += -DDBG_PRINT_OPAE
DBG_PRINT_FLAGS += -DDBG_CORE_REQ_INFO
DBG_PRINT_FLAGS += -DDBG_PRINT_SCOPE

DBG_FLAGS += $(DBG_PRINT_FLAGS)
DBG_FLAGS += -DDBG_CORE_REQ_INFO

FPU_INCLUDE = -I../rtl/fp_cores -I../rtl/fp_cores/svdpi -I../rtl/fp_cores/fpnew/src/common_cells/include -I../rtl/fp_cores/fpnew/src/common_cells/src -I../rtl/fp_cores/fpnew/src/fpu_div_sqrt_mvp/hdl -I../rtl/fp_cores/fpnew/src 
INCLUDE = -I../rtl/ -I../rtl/libs -I../rtl/interfaces -I../rtl/cache -I../rtl/simulate $(FPU_INCLUDE)

SRCS = simulator.cpp testbench.cpp
SRCS += ../rtl/fp_cores/svdpi/float_dpi.cpp

all: build-s

CF += -std=c++11 -fms-extensions -I../..

VF += --language 1800-2009 --assert -Wall -Wpedantic
VF += -Wno-DECLFILENAME
VF += --x-initial unique --x-assign unique
VF += --exe $(SRCS) $(INCLUDE)
VF += --cc Vortex.v --top-module Vortex
VF += verilator.vlt

DBG += -DVCD_OUTPUT $(DBG_FLAGS)

THREADS ?= $(shell python3 -c 'import multiprocessing as mp; print(max(1, mp.cpu_count() // 2))')

gen-s:
	verilator $(VF) -DNDEBUG  $(SINGLECORE) -CFLAGS '$(CF) -DNDEBUG $(SINGLECORE)'

gen-sd:
	verilator $(VF) -O0 $(SINGLECORE) -CFLAGS '$(CF) -O0 -g $(DBG) $(SINGLECORE)' --trace-fst --trace-threads 1 $(DBG)

gen-st:
	verilator $(VF) -DNDEBUG $(SINGLECORE) -CFLAGS '$(CF) -DNDEBUG -O2 $(SINGLECORE)' --threads $(THREADS)

gen-m:
	verilator $(VF) -DNDEBUG $(MULTICORE) -CFLAGS '$(CF) -DNDEBUG $(MULTICORE)'
	
gen-md:
	verilator $(VF) $(MULTICORE) -CFLAGS '$(CF) -O0 -g $(DBG) $(MULTICORE)' --trace-fst --trace-threads 1 $(DBG)

gen-mt:
	verilator $(VF) -DNDEBUG $(MULTICORE) -CFLAGS '$(CF) -DNDEBUG -O2 $(MULTICORE)' --threads $(THREADS)

build-s: gen-s
	(cd obj_dir && make -j -f VVortex.mk)

build-sd: gen-sd
	(cd obj_dir && OPT_FAST="-O0 -g" make -j -f VVortex.mk)

build-st: gen-st
	(cd obj_dir && make -j -f VVortex.mk)

build-m: gen-m
	(cd obj_dir && make -j -f VVortex.mk)

build-md: gen-md
	(cd obj_dir && OPT_FAST="-O0 -g" make -j -f VVortex.mk)

build-mt: gen-mt
	(cd obj_dir && make -j -f VVortex.mk)

run: run-s

run-s: build-s
	(cd obj_dir && ./VVortex)

run-sd: build-sd
	(cd obj_dir && ./VVortex)

run-st: build-st
	(cd obj_dir && ./VVortex)

run-m: build-m
	(cd obj_dir && ./VVortex)

run-md: build-md
	(cd obj_dir && ./VVortex)

run-mt: build-mt
	(cd obj_dir && ./VVortex)

clean:
	rm -rf obj_dir
