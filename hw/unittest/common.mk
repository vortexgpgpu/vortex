DESTDIR ?= .

CONFIGS +=
PARAMS +=

CXXFLAGS += -std=c++17 -Wall -Wextra -Wfatal-errors -Wno-array-bounds
CXXFLAGS += -fPIC -Wno-maybe-uninitialized
CXXFLAGS += $(CONFIGS)

LDFLAGS +=
RTL_PKGS +=
RTL_INCLUDE +=

DBG_FLAGS += -DDEBUG_LEVEL=$(DEBUG) -DVCD_OUTPUT $(DBG_TRACE_FLAGS)

VL_FLAGS = --exe
VL_FLAGS += --language 1800-2009 --assert -Wall -Wpedantic
VL_FLAGS += -Wno-DECLFILENAME -Wno-REDEFMACRO -Wno-GENUNNAMED
VL_FLAGS += --x-initial unique --x-assign unique
VL_FLAGS += -DSIMULATION -DSV_DPI
VL_FLAGS += $(CONFIGS)
VL_FLAGS += $(PARAMS)
VL_FLAGS += $(RTL_INCLUDE)
VL_FLAGS += $(RTL_PKGS)
VL_FLAGS += --cc $(TOP) --top-module $(TOP)

# Extract RTL directories from include directories
RTL_DIRS := $(patsubst -I%,%,$(filter -I%,$(RTL_INCLUDE)))

# Discover RTL source files from source directories
RTL_SRCS := $(shell find $(RTL_DIRS) -type f \( -name '*.v' -o -name '*.vh' -o -name '*.sv' \))

# Enable Verilator multithreaded simulation
THREADS ?= $(shell python3 -c 'import multiprocessing as mp; print(mp.cpu_count())')
VL_FLAGS += -j $(THREADS)
#VL_FLAGS += --threads $(THREADS)

# Debugging
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

all: $(DESTDIR)/$(PROJECT)

$(DESTDIR)/$(PROJECT): $(SRCS) $(RTL_SRCS)
	verilator --build $(VL_FLAGS) $(SRCS) -CFLAGS '$(CXXFLAGS)' --MMD -o ../$@

run: $(DESTDIR)/$(PROJECT)
	$(DESTDIR)/$(PROJECT)

waves: trace.vcd
	gtkwave -o trace.vcd

clean:
	rm -rf *.vcd obj_dir $(DESTDIR)/$(PROJECT)