all: singlecore

CF += -std=c++11 -fms-extensions

VF += -compiler gcc --language 1800-2009 --assert -Wall -Wpedantic

VF += -exe $(SRCS) $(INCLUDE)

MULTICORE += -DNUM_CLUSTERS=1 -DNUM_CORES=2

INCLUDE = -I./rtl/ -I./rtl/libs -I./rtl/interfaces -I./rtl/pipe_regs -I./rtl/cache -I./rtl/simulate

SRCS += ./simulate/testbench.cpp ./simulate/simulator.cpp

DBG += --trace -DVL_DEBUG=1 

THREADS ?= $(shell python3 -c 'import multiprocessing as mp; print(max(1, mp.cpu_count() // 2))')

.PHONY: build_config

build_config:
	./scripts/gen_config.py --outv ./rtl/VX_user_config.vh --outc ./simulate/VX_config.h

gen-singlecore: build_config
	verilator $(VF) -DNDEBUG -cc Vortex_Socket.v -CFLAGS '$(CF) -DNDEBUG'

gen-singlecore-t: build_config
	verilator $(VF) -cc Vortex_Socket.v -CFLAGS '$(CF) -DNDEBUG -O2' --threads $(THREADS)

gen-singlecore-d: build_config
	verilator $(VF) -cc Vortex_Socket.v -CFLAGS '$(CF) -g -O0 -DVCD_OUTPUT' $(DBG)

gen-multicore: build_config
	verilator $(VF) -DNDEBUG -cc Vortex_Socket.v $(MULTICORE) -CFLAGS '$(CF) -DNDEBUG $(MULTICORE)'
	
gen-multicore-t: build_config
	verilator $(VF) -DNDEBUG -cc Vortex_Socket.v $(MULTICORE) -CFLAGS '$(CF) -DNDEBUG -O2 $(MULTICORE)' --threads $(THREADS)

gen-multicore-d: build_config
	verilator $(VF) -cc Vortex_Socket.v $(MULTICORE) -CFLAGS '$(CF) -g -O0 -DVCD_OUTPUT $(MULTICORE)' $(DBG)

singlecore: gen-singlecore
	(cd obj_dir && make -j -f VVortex_Socket.mk)

singlecore-t: gen-singlecore-t
	(cd obj_dir && make -j -f VVortex_Socket.mk)

singlecore-d: gen-singlecore-d
	(cd obj_dir && make -j -f VVortex_Socket.mk)

multicore: gen-multicore
	(cd obj_dir && make -j -f VVortex_Socket.mk)

multicore-t: gen-multicore-t
	(cd obj_dir && make -j -f VVortex_Socket.mk)

multicore-d: gen-multicore-d
	(cd obj_dir && make -j -f VVortex_Socket.mk)

run: singlecore
	(cd obj_dir && ./VVortex_Socket)

run-d: singlecore-d
	(cd obj_dir && ./VVortex_Socket)

run-t: singlecore-t
	(cd obj_dir && ./VVortex_Socket)

run-m: multicore
	(cd obj_dir && ./VVortex_Socket)

run-md: multicore-d
	(cd obj_dir && ./VVortex_Socket)

run-mt: multicore-t
	(cd obj_dir && ./VVortex_Socket)

clean:
	rm -rf obj_dir