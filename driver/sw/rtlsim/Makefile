#CFLAGS += -std=c++11 -O3 -Wall -Wextra -pedantic -Wfatal-errors
CFLAGS += -std=c++11 -g -O0 -Wall -Wextra -pedantic -Wfatal-errors

#USE_MULTICORE=1

CFLAGS += -I../../include -I../../../../rtl/simulate

CFLAGS += -fPIC

CFLAGS += -DUSE_RTLSIM

LDFLAGS += -shared -pthread

ifdef USE_MULTICORE
	CFLAGS += -DUSE_MULTICORE
	RTL_TOP = Vortex_SOC
else
	RTL_TOP = Vortex
endif

SRCS = vortex.cpp ../vx_utils.cpp ../../../rtl/simulate/$(RTL_TOP).cpp

RTL_INCLUDE = -I../../../rtl -I../../../rtl/interfaces -I../../../rtl/cache -I../../../rtl/VX_cache -I../../../rtl/shared_memory -I../../../rtl/pipe_regs -I../../../rtl/compat

THREADS ?= $(shell python3 -c 'import multiprocessing as mp; print(max(1, mp.cpu_count() // 2))')
VL_FLAGS += --threads $(THREADS)

VL_FLAGS += -Wno-UNOPTFLAT -Wno-WIDTH 

VL_FLAGS += -Wno-UNDRIVEN --Wno-PINMISSING -Wno-STMTDLY -Wno-WIDTH -Wno-UNSIGNED -Wno-UNOPTFLAT -Wno-LITENDIAN

# Debugigng
VL_FLAGS += --trace -DVL_DEBUG=1 
CFLAGS += -DVCD_OUTPUT

PROJECT = libvortex.so

all: $(PROJECT)

.PHONY: build_config
build_config:
	(cd ../../../rtl && ./gen_config.py --rtl_locations)
	
$(PROJECT): $(SRCS) build_config
	verilator --exe --cc $(RTL_TOP).v $(RTL_INCLUDE) $(VL_FLAGS) $(SRCS) -CFLAGS '$(CFLAGS)' -LDFLAGS '$(LDFLAGS)' -o ../$(PROJECT)
	make -j -C obj_dir -f V$(RTL_TOP).mk

clean:
	rm -rf $(PROJECT) obj_dir