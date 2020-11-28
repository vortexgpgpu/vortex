CFLAGS += -std=c++11 -O2 -Wall -Wextra -Wfatal-errors 
#CFLAGS += -std=c++11 -g -O0 -Wall -Wextra -Wfatal-errors

CFLAGS += -Wno-aligned-new -Wno-maybe-uninitialized

CFLAGS += -I../../include -I../../../simX/include -I../../../hw

CFLAGS += -fPIC

CFLAGS += -DUSE_SIMX

LDFLAGS += -shared -pthread
#LDFLAGS += -dynamiclib -pthread

TOP = cache_simX

RTL_DIR = ../../hw/old_rtl

SRCS = vortex.cpp ../common/vx_utils.cpp 
SRCS += ../../simX/args.cpp ../../simX/mem.cpp ../../simX/core.cpp ../../simX/instruction.cpp ../../simX/enc.cpp ../../simX/util.cpp

RTL_INCLUDE = -I$(RTL_DIR) -I$(RTL_DIR)/interfaces -I$(RTL_DIR)/cache -I$(RTL_DIR)/shared_memory
RTL_INCLUDE += -I../../simX 

VL_FLAGS += -O2 --language 1800-2009 --assert
VL_FLAGS += -Wno-DECLFILENAME
VL_FLAGS += --x-initial unique --x-assign unique
VL_FLAGS += -Wno-UNOPTFLAT -Wno-WIDTH

# Enable Verilator multithreaded simulation
#THREADS ?= $(shell python3 -c 'import multiprocessing as mp; print(max(1, mp.cpu_count() // 2))')
#VL_FLAGS += --threads $(THREADS)

# Debugigng
ifdef DEBUG
	VL_FLAGS += -DVCD_OUTPUT --trace --trace-structs $(DBG_FLAGS)
	CFLAGS   += -DVCD_OUTPUT $(DBG_FLAGS)
else    
	VL_FLAGS += -DNDEBUG
	CFLAGS   += -DNDEBUG
endif

PROJECT = libvortex.so
#PROJECT = libvortex.dylib

all: $(PROJECT)

$(PROJECT): $(SRCS) 
	verilator --exe --cc $(TOP) --top-module $(TOP) $(RTL_INCLUDE) $(VL_FLAGS) $(SRCS) -CFLAGS '$(CFLAGS)' -LDFLAGS '$(LDFLAGS)' -o ../$(PROJECT)
	make -j -C obj_dir -f V$(TOP).mk

clean:
	rm -rf $(PROJECT) obj_dir