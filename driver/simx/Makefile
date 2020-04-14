CFLAGS += -std=c++11 -O3 -Wall -Wextra -pedantic -Wfatal-errors
#CFLAGS += -std=c++11 -g -O0 -Wall -Wextra -pedantic -Wfatal-errors

CFLAGS += -I../../include -I../../../simX/include -I../../../runtime

CFLAGS += -fPIC

CFLAGS += -DUSE_SIMX 

LDFLAGS += -shared -pthread

SRCS = vortex.cpp ../common/vx_utils.cpp ../../simX/args.cpp ../../simX/mem.cpp ../../simX/core.cpp ../../simX/instruction.cpp ../../simX/enc.cpp ../../simX/util.cpp

RTL_TOP = ../../simX/cache_simX.v

RTL_INCLUDE = -I../../hw/old_rtl -I../../hw/old_rtl/interfaces -I../../hw/old_rtl/cache -I../../hw/old_rtl/shared_memory

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