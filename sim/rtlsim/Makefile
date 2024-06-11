include ../common.mk

DESTDIR ?= $(CURDIR)

SRC_DIR = $(VORTEX_HOME)/sim/rtlsim

CXXFLAGS += -std=c++11 -Wall -Wextra -Wfatal-errors -Wno-array-bounds
CXXFLAGS += -fPIC -Wno-maybe-uninitialized
CXXFLAGS += -I$(ROOT_DIR)/hw -I$(COMMON_DIR)
CXXFLAGS += -I$(THIRD_PARTY_DIR)/softfloat/source/include
CXXFLAGS += -I$(THIRD_PARTY_DIR)
CXXFLAGS += -DXLEN_$(XLEN)

LDFLAGS += $(THIRD_PARTY_DIR)/softfloat/build/Linux-x86_64-GCC/softfloat.a
LDFLAGS += -L$(THIRD_PARTY_DIR)/ramulator -lramulator

# control RTL debug tracing states
DBG_TRACE_FLAGS += -DDBG_TRACE_PIPELINE
DBG_TRACE_FLAGS += -DDBG_TRACE_MEM
DBG_TRACE_FLAGS += -DDBG_TRACE_CACHE
DBG_TRACE_FLAGS += -DDBG_TRACE_AFU
DBG_TRACE_FLAGS += -DDBG_TRACE_SCOPE
DBG_TRACE_FLAGS += -DDBG_TRACE_GBAR

DBG_FLAGS += -DDEBUG_LEVEL=$(DEBUG) -DVCD_OUTPUT $(DBG_TRACE_FLAGS)

RTL_PKGS = $(RTL_DIR)/VX_gpu_pkg.sv $(RTL_DIR)/fpu/VX_fpu_pkg.sv

FPU_INCLUDE = -I$(RTL_DIR)/fpu
ifneq (,$(findstring FPU_FPNEW,$(CONFIGS)))
	RTL_PKGS += $(THIRD_PARTY_DIR)/fpnew/src/fpnew_pkg.sv $(THIRD_PARTY_DIR)/fpnew/src/common_cells/src/cf_math_pkg $(THIRD_PARTY_DIR)/fpnew/src/fpu_div_sqrt_mvp/hdl/defs_div_sqrt_mvp.sv
	FPU_INCLUDE += -I$(THIRD_PARTY_DIR)/fpnew/src/common_cells/include -I$(THIRD_PARTY_DIR)/fpnew/src/common_cells/src -I$(THIRD_PARTY_DIR)/fpnew/src/fpu_div_sqrt_mvp/hdl -I$(THIRD_PARTY_DIR)/fpnew/src
endif
RTL_INCLUDE = -I$(RTL_DIR) -I$(DPI_DIR) -I$(RTL_DIR)/libs -I$(RTL_DIR)/interfaces -I$(RTL_DIR)/core -I$(RTL_DIR)/mem -I$(RTL_DIR)/cache $(FPU_INCLUDE)

SRCS = $(COMMON_DIR)/util.cpp $(COMMON_DIR)/mem.cpp $(COMMON_DIR)/rvfloats.cpp
SRCS += $(DPI_DIR)/util_dpi.cpp $(DPI_DIR)/float_dpi.cpp
SRCS += $(SRC_DIR)/processor.cpp

ifdef AXI_BUS
	TOP = Vortex_axi
	CXXFLAGS += -DAXI_BUS
else
	TOP = Vortex
endif

VL_FLAGS = --exe
VL_FLAGS += --language 1800-2009 --assert -Wall -Wpedantic
VL_FLAGS += -Wno-DECLFILENAME -Wno-REDEFMACRO
VL_FLAGS += --x-initial unique --x-assign unique
VL_FLAGS += $(SRC_DIR)/verilator.vlt
VL_FLAGS += -DSIMULATION -DSV_DPI
VL_FLAGS += -DXLEN_$(XLEN)
VL_FLAGS += $(CONFIGS)
VL_FLAGS += $(RTL_INCLUDE)
VL_FLAGS += $(RTL_PKGS)
VL_FLAGS += --cc $(TOP) --top-module $(TOP)

CXXFLAGS += $(CONFIGS)

# Enable Verilator multithreaded simulation
THREADS ?= $(shell python -c 'import multiprocessing as mp; print(mp.cpu_count())')
VL_FLAGS += -j $(THREADS)
#VL_FLAGS += --threads $(THREADS)

# Debugigng
ifdef DEBUG
	VL_FLAGS += --trace --trace-structs $(DBG_FLAGS)
	CXXFLAGS += -g -O0 $(DBG_FLAGS)
else
	VL_FLAGS += -DNDEBUG
	CXXFLAGS += -O2 -DNDEBUG
endif

# Enable perf counters
ifdef PERF
	VL_FLAGS += -DPERF_ENABLE
	CXXFLAGS += -DPERF_ENABLE
endif

PROJECT := rtlsim

all: $(DESTDIR)/$(PROJECT)

$(DESTDIR)/$(PROJECT): $(SRCS) $(SRC_DIR)/main.cpp
	verilator --build $(VL_FLAGS) $^ -CFLAGS '$(CXXFLAGS) -DSTARTUP_ADDR=0x80000000' -LDFLAGS '$(LDFLAGS)' --Mdir $@.obj_dir -o $@

$(DESTDIR)/lib$(PROJECT).so: $(SRCS)
	verilator --build $(VL_FLAGS) $^ -CFLAGS '$(CXXFLAGS)' -LDFLAGS '-shared $(LDFLAGS)' --Mdir $@.obj_dir -o $@

clean-lib:
	rm -rf $(DESTDIR)/lib$(PROJECT).so.obj_dir
	rm -f $(DESTDIR)/lib$(PROJECT).so

clean-exe:
	rm -rf $(DESTDIR)/$(PROJECT).obj_dir
	rm -f $(DESTDIR)/$(PROJECT)

clean: clean-lib clean-exe