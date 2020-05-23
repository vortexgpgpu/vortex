all: build-s

CF += -std=c++11 -fms-extensions

VF += --language 1800-2009 --assert -Wall -Wpedantic
VF += -Wno-DECLFILENAME
VF += --x-initial unique

VF += -exe $(SRCS) $(INCLUDE)

# Use 64 bytes DRAM blocks
CF += -DGLOBAL_BLOCK_SIZE=64
VF += -DGLOBAL_BLOCK_SIZE=64

#MULTICORE += -DNUM_CLUSTERS=2 -DNUM_CORES=4
#MULTICORE += -DNUM_CLUSTERS=1 -DNUM_CORES=4
MULTICORE += -DNUM_CLUSTERS=1 -DNUM_CORES=2

# control RTL debug print states
DBG_PRINT_FLAGS = -DDBG_PRINT_CORE_ICACHE \
				  -DDBG_PRINT_CORE_DCACHE \
			      -DDBG_PRINT_CACHE_BANK  \
				  -DDBG_PRINT_CACHE_SNP   \
				  -DDBG_PRINT_CACHE_MSRQ  \
				  -DDBG_PRINT_DRAM        \
				  -DDBG_PRINT_OPAE

#DBG_PRINT=$(DBG_PRINT_FLAGSs)

INCLUDE = -I./rtl/ -I./rtl/libs -I./rtl/interfaces -I./rtl/pipe_regs -I./rtl/cache -I./rtl/simulate

SRCS += ./simulate/testbench.cpp ./simulate/simulator.cpp

DBG += --trace

THREADS ?= $(shell python3 -c 'import multiprocessing as mp; print(max(1, mp.cpu_count() // 2))')

.PHONY: build_config

build_config:
	./scripts/gen_config.py --outv ./rtl/VX_user_config.vh --outc ./simulate/VX_config.h

gen-s: build_config
	verilator $(VF) -DNDEBUG -cc Vortex_Socket.v -CFLAGS '$(CF) -DNDEBUG'

gen-sd: build_config
	verilator $(VF) -cc Vortex_Socket.v $(DBG_PRINT) -CFLAGS '$(CF) -g -O0 -DVCD_OUTPUT' $(DBG)

gen-st: build_config
	verilator $(VF) -DNDEBUG -cc Vortex_Socket.v -CFLAGS '$(CF) -DNDEBUG -O2' --threads $(THREADS)

gen-m: build_config
	verilator $(VF) -DNDEBUG -cc Vortex_Socket.v $(MULTICORE) -CFLAGS '$(CF) -DNDEBUG $(MULTICORE)'
	
gen-md: build_config
	verilator $(VF) -cc Vortex_Socket.v $(MULTICORE) $(DBG_PRINT) -CFLAGS '$(CF) -g -O0 -DVCD_OUTPUT $(MULTICORE)' $(DBG)

gen-mt: build_config
	verilator $(VF) -DNDEBUG -cc Vortex_Socket.v $(MULTICORE) -CFLAGS '$(CF) -DNDEBUG -O2 $(MULTICORE)' --threads $(THREADS)

build-s: gen-s
	(cd obj_dir && make -j -f VVortex_Socket.mk)

build-sd: gen-sd
	(cd obj_dir && make -j -f VVortex_Socket.mk)

build-st: gen-st
	(cd obj_dir && make -j -f VVortex_Socket.mk)

build-m: gen-m
	(cd obj_dir && make -j -f VVortex_Socket.mk)

build-md: gen-md
	(cd obj_dir && make -j -f VVortex_Socket.mk)

build-mt: gen-mt
	(cd obj_dir && make -j -f VVortex_Socket.mk)

run: run-s
run-s: build-s
	(cd obj_dir && ./VVortex_Socket)

run-sd: build-sd
	(cd obj_dir && ./VVortex_Socket)

run-st: build-st
	(cd obj_dir && ./VVortex_Socket)

run-m: build-m
	(cd obj_dir && ./VVortex_Socket)

run-md: build-md
	(cd obj_dir && ./VVortex_Socket)

run-mt: build-mt
	(cd obj_dir && ./VVortex_Socket)

clean:
	rm -rf obj_dir