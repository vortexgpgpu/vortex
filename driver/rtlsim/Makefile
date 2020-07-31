#CFLAGS += -std=c++11 -O2 -DNDEBUG -Wall -Wextra -Wfatal-errors
CFLAGS += -std=c++11 -g -O0 -Wall -Wextra -Wfatal-errors

CFLAGS += -I../../include -I../../../hw/simulate -I../../../hw

# control RTL debug print states
DBG_PRINT_FLAGS += -DDBG_PRINT_CORE_ICACHE
DBG_PRINT_FLAGS += -DDBG_PRINT_CORE_DCACHE
DBG_PRINT_FLAGS += -DDBG_PRINT_CACHE_BANK 
DBG_PRINT_FLAGS += -DDBG_PRINT_CACHE_SNP 
DBG_PRINT_FLAGS += -DDBG_PRINT_CACHE_MSRQ
DBG_PRINT_FLAGS += -DDBG_PRINT_DRAM      
DBG_PRINT_FLAGS += -DDBG_PRINT_PIPELINE  
DBG_PRINT_FLAGS += -DDBG_PRINT_OPAE

DBG_FLAGS += $(DBG_PRINT_FLAGS)
DBG_FLAGS += -DDBG_CORE_REQ_INFO

#CONFIGS += -DNUM_CLUSTERS=2 -DNUM_CORES=4 -DL2_ENABLE=1
#CONFIGS += -DNUM_CLUSTERS=1 -DNUM_CORES=4 -DL2_ENABLE=1
#CONFIGS += -DNUM_CLUSTERS=1 -DNUM_CORES=2 -DL2_ENABLE=0
CONFIGS += -DNUM_CLUSTERS=1 -DNUM_CORES=1

#DEBUG=1
#AFU=1

CFLAGS += -fPIC

CFLAGS += -DUSE_RTLSIM $(CONFIGS)

LDFLAGS += -shared -pthread
# LDFLAGS += -dynamiclib -pthread

TOP = Vortex

SRCS = vortex.cpp ../common/vx_utils.cpp ../../hw/simulate/simulator.cpp

FPU_INCLUDE = -I../../hw/rtl/fp_cores/fpnew/src/common_cells/include -I../../hw/rtl/fp_cores/fpnew/src/common_cells/src -I../../hw/rtl/fp_cores/fpnew/src/fpu_div_sqrt_mvp/hdl -I../../hw/rtl/fp_cores/fpnew/src 
RTL_INCLUDE = -I../../hw/rtl -I../../hw/rtl/libs -I../../hw/rtl/interfaces -I../../hw/rtl/cache -I../../hw/rtl/fp_cores $(FPU_INCLUDE)

VL_FLAGS += --language 1800-2009 --assert -Wall -Wpedantic $(CONFIGS)
VL_FLAGS += -Wno-DECLFILENAME
VL_FLAGS += --x-initial unique --x-assign unique
VL_FLAGS += verilator.vlt

# Enable Verilator multithreaded simulation
#THREADS ?= $(shell python3 -c 'import multiprocessing as mp; print(max(1, mp.cpu_count() // 2))')
#VL_FLAGS += --threads $(THREADS)

# Debugigng
ifdef DEBUG
	VL_FLAGS += -DVCD_OUTPUT --assert --trace $(DBG_FLAGS)
	CFLAGS   += -DVCD_OUTPUT $(DBG_FLAGS)
else    
	VL_FLAGS += -DNDEBUG
	CFLAGS   += -DNDEBUG
endif

# AFU
ifdef AFU
    TOP = vortex_afu_sim
	VL_FLAGS += -DNOPAE
	CFLAGS += -DNOPAE
	#VL_FLAGS += -DSCOPE
	#CFLAGS += -DSCOPE
	RTL_INCLUDE += -I../../hw/opae  -I../../hw/opae/ccip
endif

PROJECT = libvortex.so
# PROJECT = libvortex.dylib

all: $(PROJECT)
	
$(PROJECT): $(SRCS)
	verilator --exe --cc $(TOP) --top-module $(TOP) $(RTL_INCLUDE) $(VL_FLAGS) $(SRCS) -CFLAGS '$(CFLAGS)' -LDFLAGS '$(LDFLAGS)' -o ../$(PROJECT)
	make -j -C obj_dir -f V$(TOP).mk

clean:
	rm -rf $(PROJECT) obj_dir
