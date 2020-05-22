#CFLAGS += -std=c++11 -O2 -DNDEBUG -Wall -Wextra -Wfatal-errors
CFLAGS += -std=c++11 -g -O0 -Wall -Wextra -Wfatal-errors

CFLAGS += -I../../include -I../../../hw/simulate -I../../../runtime

# control RTL debug print states
DBG_PRINT = -DDBG_PRINT_CORE_ICACHE \
            -DDBG_PRINT_CORE_DCACHE \
            -DDBG_PRINT_CACHE_BANK  \
            -DDBG_PRINT_CACHE_SNP   \
			-DDBG_PRINT_CACHE_MSRQ  \
			-DDBG_PRINT_DRAM        \
			-DDBG_PRINT_OPAE

#MULTICORE += -DNUM_CLUSTERS=2 -DNUM_CORES=4
#MULTICORE += -DNUM_CLUSTERS=1 -DNUM_CORES=4
MULTICORE += -DNUM_CLUSTERS=1 -DNUM_CORES=2

#DEBUG = 1

CFLAGS += -fPIC

CFLAGS += -DUSE_RTLSIM $(MULTICORE)

LDFLAGS += -shared -pthread
# LDFLAGS += -dynamiclib -pthread

SRCS = vortex.cpp ../common/vx_utils.cpp ../../hw/simulate/simulator.cpp

RTL_INCLUDE = -I../../hw/rtl -I../../hw/rtl/libs -I../../hw/rtl/interfaces -I../../hw/rtl/pipe_regs -I../../hw/rtl/cache

VL_FLAGS += --language 1800-2009 --assert -Wall -Wpedantic $(MULTICORE)
VL_FLAGS += -Wno-DECLFILENAME
VL_FLAGS += --x-initial unique

# Use 64 bytes DRAM blocks
CFLAGS += -DGLOBAL_BLOCK_SIZE=64
VL_FLAGS += -DGLOBAL_BLOCK_SIZE=64

# Enable Verilator multithreaded simulation
#THREADS ?= $(shell python3 -c 'import multiprocessing as mp; print(max(1, mp.cpu_count() // 2))')
#VL_FLAGS += --threads $(THREADS)

# Debugigng
ifdef DEBUG
	VL_FLAGS += --trace -DVL_DEBUG=1 $(DBG_PRINT)
	CFLAGS += -DVCD_OUTPUT
else
    CFLAGS += -DNDEBUG
	VL_FLAGS += -DNDEBUG
endif

PROJECT = libvortex.so
# PROJECT = libvortex.dylib

all: $(PROJECT)
	
$(PROJECT): $(SRCS)
	verilator --exe --cc Vortex_Socket.v $(RTL_INCLUDE) $(VL_FLAGS) $(SRCS) -CFLAGS '$(CFLAGS)' -LDFLAGS '$(LDFLAGS)' -o ../$(PROJECT)
	make -j -C obj_dir -f VVortex_Socket.mk

clean:
	rm -rf $(PROJECT) obj_dir
