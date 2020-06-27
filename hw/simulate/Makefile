#MULTICORE += -DNUM_CLUSTERS=2 -DNUM_CORES=4
#MULTICORE += -DNUM_CLUSTERS=1 -DNUM_CORES=4
MULTICORE += -DNUM_CLUSTERS=1 -DNUM_CORES=2

# control RTL debug print states
DBG_PRINT_FLAGS += -DDBG_PRINT_CORE_ICACHE
DBG_PRINT_FLAGS += -DDBG_PRINT_CORE_DCACHE
DBG_PRINT_FLAGS += -DDBG_PRINT_CACHE_BANK 
DBG_PRINT_FLAGS += -DDBG_PRINT_CACHE_SNP 
DBG_PRINT_FLAGS += -DDBG_PRINT_CACHE_MSRQ
DBG_PRINT_FLAGS += -DDBG_PRINT_DRAM      
DBG_PRINT_FLAGS += -DDBG_PRINT_PIPELINE  
DBG_PRINT_FLAGS += -DDBG_PRINT_OPAE

DBG_FLAGS += $(DBG_PRINT_FLAGS)
DBG_FLAGS += -DDBG_CORE_REQ_INFO

INCLUDE = -I../rtl/ -I../rtl/libs -I../rtl/interfaces -I../rtl/pipe_regs -I../rtl/cache -I../rtl/simulate

SRCS = simulator.cpp testbench.cpp

all: build-s

CF += -std=c++11 -fms-extensions -I../..

VF += --language 1800-2009 --assert -Wall -Wpedantic
VF += -Wno-DECLFILENAME
VF += --x-initial unique
VF += -exe $(SRCS) $(INCLUDE)

DBG += -DVCD_OUTPUT $(DBG_FLAGS)
DBG += -DDBG_CORE_REQ_INFO

THREADS ?= $(shell python3 -c 'import multiprocessing as mp; print(max(1, mp.cpu_count() // 2))')

gen-s:
	verilator $(VF) -DNDEBUG -cc Vortex.v -CFLAGS '$(CF) -DNDEBUG'

gen-sd:
	verilator $(VF) -cc Vortex.v -CFLAGS '$(CF) -g -O0  $(DBG)' --trace $(DBG)

gen-st:
	verilator $(VF) -DNDEBUG -cc Vortex.v -CFLAGS '$(CF) -DNDEBUG -O2' --threads $(THREADS)

gen-m:
	verilator $(VF) -DNDEBUG -cc Vortex.v $(MULTICORE) -CFLAGS '$(CF) -DNDEBUG $(MULTICORE)'
	
gen-md:
	verilator $(VF) -cc Vortex.v $(MULTICORE) -CFLAGS '$(CF) -g -O0 $(DBG) $(MULTICORE)' --trace $(DBG)

gen-mt:
	verilator $(VF) -DNDEBUG -cc Vortex.v $(MULTICORE) -CFLAGS '$(CF) -DNDEBUG -O2 $(MULTICORE)' --threads $(THREADS)

build-s: gen-s
	(cd obj_dir && make -j -f VVortex.mk)

build-sd: gen-sd
	(cd obj_dir && make -j -f VVortex.mk)

build-st: gen-st
	(cd obj_dir && make -j -f VVortex.mk)

build-m: gen-m
	(cd obj_dir && make -j -f VVortex.mk)

build-md: gen-md
	(cd obj_dir && make -j -f VVortex.mk)

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