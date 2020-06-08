#CFLAGS += -std=c++11 -O2 -DNDEBUG -Wall -Wextra -Wfatal-errors
CFLAGS += -std=c++11 -g -O0 -Wall -Wextra -Wfatal-errors

CFLAGS += -I../../include -I../../../hw/simulate -I../../../hw

# control RTL debug print states
DBG_PRINT_FLAGS = -DDBG_PRINT_CORE_ICACHE \
				  -DDBG_PRINT_CORE_DCACHE \
			      -DDBG_PRINT_CACHE_BANK  \
				  -DDBG_PRINT_CACHE_SNP   \
				  -DDBG_PRINT_CACHE_MSRQ  \
				  -DDBG_PRINT_DRAM        \
				  -DDBG_PRINT_OPAE

#DBG_PRINT=$(DBG_PRINT_FLAGS)

#MULTICORE += -DNUM_CLUSTERS=2 -DNUM_CORES=4
#MULTICORE += -DNUM_CLUSTERS=1 -DNUM_CORES=4
#MULTICORE += -DNUM_CLUSTERS=1 -DNUM_CORES=2

#DEBUG=1
AFU=1

CFLAGS += -fPIC

CFLAGS += -DUSE_RTLSIM $(MULTICORE)

LDFLAGS += -shared -pthread
# LDFLAGS += -dynamiclib -pthread

TOP = Vortex_Socket

SRCS = vortex.cpp ../common/vx_utils.cpp ../../hw/simulate/simulator.cpp

RTL_INCLUDE = -I../../hw/rtl -I../../hw/rtl/libs -I../../hw/rtl/interfaces -I../../hw/rtl/pipe_regs -I../../hw/rtl/cache

VL_FLAGS += --language 1800-2009 --assert -Wall -Wpedantic $(MULTICORE)
VL_FLAGS += -Wno-DECLFILENAME
VL_FLAGS += --x-initial unique
VL_FLAGS += --x-assign unique

# Enable Verilator multithreaded simulation
#THREADS ?= $(shell python3 -c 'import multiprocessing as mp; print(max(1, mp.cpu_count() // 2))')
#VL_FLAGS += --threads $(THREADS)

# Debugigng
ifdef DEBUG
	VL_FLAGS += --trace -DVCD_OUTPUT $(DBG_PRINT)
	CFLAGS += -DVCD_OUTPUT $(DBG_PRINT)
else
    CFLAGS += -DNDEBUG
	VL_FLAGS += -DNDEBUG
endif

# AFU
ifdef AFU
    TOP = vortex_afu_sim
	VL_FLAGS += -DNOPAE -DSCOPE
	CFLAGS += -DNOPAE -DSCOPE
	RTL_INCLUDE += -I../../hw/opae  -I../../hw/opae/ccip
endif

PROJECT = libvortex.so
# PROJECT = libvortex.dylib

all: $(PROJECT)
	
$(PROJECT): $(SRCS)
	verilator --exe --cc $(TOP) $(RTL_INCLUDE) $(VL_FLAGS) $(SRCS) -CFLAGS '$(CFLAGS)' -LDFLAGS '$(LDFLAGS)' -o ../$(PROJECT)
	make -j -C obj_dir -f V$(TOP).mk

clean:
	rm -rf $(PROJECT) obj_dir
