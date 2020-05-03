all: singlecore

CF += -std=c++11 -fms-extensions

VF += --language 1800-2009 --assert -Wall -Wpedantic

VF += -exe $(SRCS) $(INCLUDE)

MULTICORE += -DNUM_CLUSTERS=1 -DNUM_CORES=2 -DL2_ENABLE=0

INCLUDE = -I./rtl/ -I./rtl/libs -I./rtl/interfaces -I./rtl/pipe_regs -I./rtl/cache -I./rtl/simulate

SRCS += ./simulate/testbench.cpp ./simulate/simulator.cpp

DBG += --trace -DVL_DEBUG=1 

THREADS ?= $(shell python3 -c 'import multiprocessing as mp; print(max(1, mp.cpu_count() // 2))')

.PHONY: build_config

build_config:
	./scripts/gen_config.py --outv ./rtl/VX_user_config.vh --outc ./simulate/VX_config.h

gen-s: build_config
	verilator $(VF) -DNDEBUG -cc Vortex_Socket.v -CFLAGS '$(CF) -DNDEBUG'

gen-sd: build_config
	verilator $(VF) -cc Vortex_Socket.v -CFLAGS '$(CF) -g -O0 -DVCD_OUTPUT' $(DBG)

gen-st: build_config
	verilator $(VF) -cc Vortex_Socket.v -CFLAGS '$(CF) -DNDEBUG -O2' --threads $(THREADS)

gen-m: build_config
	verilator $(VF) -DNDEBUG -cc Vortex_Socket.v $(MULTICORE) -CFLAGS '$(CF) -DNDEBUG $(MULTICORE)'
	
gen-md: build_config
	verilator $(VF) -cc Vortex_Socket.v $(MULTICORE) -CFLAGS '$(CF) -g -O0 -DVCD_OUTPUT $(MULTICORE)' $(DBG)

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