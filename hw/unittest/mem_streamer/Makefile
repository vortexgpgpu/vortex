DESTDIR ?= .
RTL_DIR = ../../rtl
DPI_DIR = ../../dpi

CONFIGS +=
PARAMS +=

CXXFLAGS += -std=c++11 -Wall -Wextra -Wfatal-errors -Wno-array-bounds
CXXFLAGS += -fPIC -Wno-maybe-uninitialized
CXXFLAGS += -I../../.. -I../../common -I../../../../sim/common
CXXFLAGS += $(CONFIGS)

LDFLAGS +=

DBG_FLAGS += -DDEBUG_LEVEL=$(DEBUG) -DVCD_OUTPUT $(DBG_TRACE_FLAGS)

RTL_PKGS +=

RTL_INCLUDE = -I$(RTL_DIR) -I$(DPI_DIR) -I$(RTL_DIR)/libs

SRCS = memsim.cpp ram.cpp
SRCS += $(DPI_DIR)/util_dpi.cpp

TOP = VX_mem_scheduler

VL_FLAGS = --exe
VL_FLAGS += --language 1800-2009 --assert -Wall -Wpedantic
VL_FLAGS += -Wno-DECLFILENAME -Wno-REDEFMACRO
VL_FLAGS += --x-initial unique --x-assign unique
VL_FLAGS += -DSIMULATION -DSV_DPI
VL_FLAGS += $(CONFIGS)
VL_FLAGS += $(PARAMS)
VL_FLAGS += $(RTL_INCLUDE)
VL_FLAGS += $(RTL_PKGS)
VL_FLAGS += --cc $(TOP) --top-module $(TOP)

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

PROJECT = mem_streamer

all: $(DESTDIR)/$(PROJECT)
	
$(DESTDIR)/$(PROJECT): $(SRCS)
	verilator --build $(VL_FLAGS) $^ -CFLAGS '$(CXXFLAGS)' -o ../$@

run: $(DESTDIR)/$(PROJECT)
	$(DESTDIR)/$(PROJECT)

waves: trace.vcd
	gtkwave -o trace.vcd

clean:
	rm -rf obj_dir $(DESTDIR)/$(PROJECT)
