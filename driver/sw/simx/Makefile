CFLAGS += -std=c++11 -O3 -Wall -Wextra -pedantic -Wfatal-errors
#CFLAGS += -std=c++11 -g -O0 -Wall -Wextra -pedantic -Wfatal-errors

MAX_WARPS ?= 8
MAX_THREADS ?= 4

CFLAGS += -I../../include -I../../../../simX/include

CFLAGS += -fPIC

CFLAGS += -DUSE_SIMX 

CFLAGS += -DMAX_WARPS=$(MAX_WARPS) -DMAX_THREADS=$(MAX_THREADS)

LDFLAGS += -shared -pthread

SRCS = vortex.cpp ../vx_utils.cpp ../../../simX/args.cpp ../../../simX/mem.cpp ../../../simX/core.cpp ../../../simX/instruction.cpp ../../../simX/enc.cpp ../../../simX/util.cpp

RTL_TOP = ../../../simX/cache_simX.v

RTL_INCLUDE = -I../../../old_rtl -I../../../old_rtl/interfaces -I../../../old_rtl/cache -I../../../old_rtl/shared_memory

THREADS ?= $(shell python3 -c 'import multiprocessing as mp; print(max(1, mp.cpu_count() // 2))')
VL_FLAGS += --threads $(THREADS)

VL_FLAGS += -Wno-UNOPTFLAT -Wno-WIDTH 

PROJECT = libvortex.so

all: $(PROJECT)

$(PROJECT): $(SRCS) 
	verilator --exe --cc $(RTL_TOP) $(RTL_INCLUDE) $(VL_FLAGS) $(SRCS) -CFLAGS '$(CFLAGS)' -LDFLAGS '$(LDFLAGS)' -o ../$(PROJECT)
	make -j -C obj_dir -f Vcache_simX.mk

clean:
	rm -rf $(PROJECT) obj_dir