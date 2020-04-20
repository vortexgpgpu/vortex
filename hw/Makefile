all: singlecore

CF += -std=c++11 -fms-extensions

VF += -compiler gcc --language 1800-2009 --assert -Wall -Wpedantic

INCLUDE = -I./rtl/ -I./rtl/libs -I./rtl/interfaces -I./rtl/pipe_regs -I./rtl/cache -I./rtl/simulate

SINGLE_CORE = Vortex.v

MULTI_CORE = Vortex_Socket.v

SRCS += ./simulate/testbench.cpp ./simulate/simulator.cpp

VF += -exe $(SRCS) $(INCLUDE)

DBG += --trace -DVL_DEBUG=1 

MAKECPP_S += (cd obj_dir && make -j -f VVortex.mk)

MAKECPP_M += (cd obj_dir && make -j -f VVortex_Socket.mk)

THREADS ?= $(shell python3 -c 'import multiprocessing as mp; print(max(1, mp.cpu_count() // 2))')

.PHONY: build_config

build_config:
	./scripts/gen_config.py --outv ./rtl/VX_user_config.vh --outc ./simulate/VX_config.h

gen-singlecore: build_config
	verilator $(VF) -DNDEBUG -cc $(SINGLE_CORE) -CFLAGS '$(CF) -DNDEBUG'

gen-singlecore-t: build_config
	verilator $(VF) -cc $(SINGLE_CORE) -CFLAGS '$(CF) -DNDEBUG -O2' --threads $(THREADS)

gen-singlecore-d: build_config
	verilator $(VF) -cc $(SINGLE_CORE) -CFLAGS '$(CF) -g -O0 -DVCD_OUTPUT' $(DBG)

gen-multicore: build_config
	verilator $(VF) -DNDEBUG -cc $(MULTI_CORE) -CFLAGS '$(CF) -DNDEBUG -DUSE_MULTICORE'
	
gen-multicore-t: build_config
	verilator $(VF) -DNDEBUG -cc $(MULTI_CORE) -CFLAGS '$(CF) -DNDEBUG -O2 -DUSE_MULTICORE' --threads $(THREADS)

gen-multicore-d: build_config
	verilator $(VF) -cc $(MULTI_CORE) -CFLAGS '$(CF) -g -O0 -DVCD_OUTPUT -DUSE_MULTICORE' $(DBG)

singlecore: gen-singlecore
	(cd obj_dir && make -j -f VVortex.mk)

singlecore-t: gen-singlecore-t
	(cd obj_dir && make -j -f VVortex.mk)

singlecore-d: gen-singlecore-d
	(cd obj_dir && make -j -f VVortex.mk)

multicore: gen-multicore
	(cd obj_dir && make -j -f VVortex_Socket.mk)

multicore-t: gen-multicore-t
	(cd obj_dir && make -j -f VVortex_Socket.mk)

multicore-d: gen-multicore-d
	(cd obj_dir && make -j -f VVortex_Socket.mk)

run: singlecore
	(cd obj_dir && ./VVortex)

run-d: singlecore-d
	(cd obj_dir && ./VVortex)

run-t: singlecore-t
	(cd obj_dir && ./VVortex)

run-m: multicore
	(cd obj_dir && ./VVortex_Socket)

run-md: multicore-d
	(cd obj_dir && ./VVortex_Socket)

run-mt: multicore-t
	(cd obj_dir && ./VVortex_Socket)

clean:
	rm -rf obj_dir