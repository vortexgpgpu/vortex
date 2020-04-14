# CFLAGS += -std=c++11 -O3 -Wall -Wextra -pedantic -Wfatal-errors
CFLAGS += -std=c++11 -O2 -Wall -Wextra -pedantic -Wfatal-errors
# CFLAGS += -std=c++11 -g -O0 -Wall -Wextra -pedantic -Wfatal-errors

USE_MULTICORE=1

CFLAGS += -I../../include -I../../../hw/simulate -I../../../runtime

CFLAGS += -fPIC

CFLAGS += -DUSE_RTLSIM

LDFLAGS += -shared -pthread

ifdef USE_MULTICORE
	CFLAGS += -DUSE_MULTICORE
	RTL_TOP = Vortex_Socket
else
    VL_FLAGS += -DSINGLE_CORE_BENCH
	RTL_TOP = Vortex
endif

SRCS = vortex.cpp ../common/vx_utils.cpp ../../hw/simulate/simulator.cpp

RTL_INCLUDE = -I../../hw/rtl -I../../hw/rtl/interfaces -I../../hw/rtl/cache -I../../hw/rtl/generic_cache -I../../hw/rtl/shared_memory -I../../hw/rtl/pipe_regs -I../../hw/rtl/compat

# Enable Verilator multithreaded simulation
#THREADS ?= $(shell python3 -c 'import multiprocessing as mp; print(max(1, mp.cpu_count() // 2))')
#VL_FLAGS += --threads $(THREADS)

VL_FLAGS += -Wno-UNDRIVEN --Wno-PINMISSING -Wno-STMTDLY -Wno-WIDTH -Wno-UNSIGNED -Wno-UNOPTFLAT -Wno-LITENDIAN -Wno-BLKLOOPINIT

# Debugigng
#VL_FLAGS += --trace -DVL_DEBUG=1 
#CFLAGS += -DVCD_OUTPUT

PROJECT = libvortex.so

all: $(PROJECT)
	
$(PROJECT): $(SRCS)
	verilator --exe --cc $(RTL_TOP).v $(RTL_INCLUDE) $(VL_FLAGS) $(SRCS) -CFLAGS '$(CFLAGS)' -LDFLAGS '$(LDFLAGS)' -o ../$(PROJECT)
	make -j -C obj_dir -f V$(RTL_TOP).mk

clean:
	rm -rf $(PROJECT) obj_dir
