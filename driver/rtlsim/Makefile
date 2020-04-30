CFLAGS += -std=c++11 -O2 -DNDEBUG -Wall -Wextra -Wfatal-errors
# CFLAGS += -std=c++11 -g -O0 -Wall -Wextra -Wfatal-errors

CFLAGS += -I../../include -I../../../hw/simulate -I../../../runtime

MULTICORE += -DNUM_CLUSTERS=2 -DNUM_CORES=2
#DEBUG = 1

CFLAGS += -fPIC

CFLAGS += -DUSE_RTLSIM $(MULTICORE)

LDFLAGS += -shared -pthread

SRCS = vortex.cpp ../common/vx_utils.cpp ../../hw/simulate/simulator.cpp

RTL_INCLUDE = -I../../hw/rtl -I../../hw/rtl/libs -I../../hw/rtl/interfaces -I../../hw/rtl/pipe_regs -I../../hw/rtl/cache

VL_FLAGS += --assert -Wall -Wpedantic $(MULTICORE)

# Enable Verilator multithreaded simulation
#THREADS ?= $(shell python3 -c 'import multiprocessing as mp; print(max(1, mp.cpu_count() // 2))')
#VL_FLAGS += --threads $(THREADS)

# Debugigng
ifdef DEBUG
	VL_FLAGS += --trace -DVL_DEBUG=1 
	CFLAGS += -DVCD_OUTPUT
else
    CFLAGS += -DNDEBUG
	VL_FLAGS += -DNDEBUG
endif

PROJECT = libvortex.so

all: $(PROJECT)
	
$(PROJECT): $(SRCS)
	verilator --exe --cc Vortex_Socket.v $(RTL_INCLUDE) $(VL_FLAGS) $(SRCS) -CFLAGS '$(CFLAGS)' -LDFLAGS '$(LDFLAGS)' -o ../$(PROJECT)
	make -j -C obj_dir -f VVortex_Socket.mk

clean:
	rm -rf $(PROJECT) obj_dir
