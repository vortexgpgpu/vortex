DESTDIR ?= .

CONFIGS +=
PARAMS +=

CXXFLAGS += -std=c++17 -Wall -Wextra -Wfatal-errors -Werror
CXXFLAGS += -fPIC
CXXFLAGS += -DVX_CFG_XLEN=$(XLEN) -DVX_CFG_XLEN_$(XLEN)
CXXFLAGS += $(CONFIGS)

LDFLAGS +=
RTL_PKGS +=
RTL_INCLUDE +=

DBG_FLAGS += -DVX_DBG_DEBUG_LEVEL=$(DEBUG) $(DBG_TRACE_FLAGS)

VL_FLAGS += --exe
VL_FLAGS += --language 1800-2012 --assert -Wall -Wpedantic
VL_FLAGS += -Wno-DECLFILENAME -Wno-REDEFMACRO -Wno-GENUNNAMED
VL_FLAGS += --x-initial unique --x-assign unique
VL_FLAGS += ../verilator.vlt
VL_FLAGS += -DSIMULATION
VL_FLAGS += -DVX_CFG_XLEN=$(XLEN) -DVX_CFG_XLEN_$(XLEN)
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
	VL_FLAGS += $(DBG_FLAGS)
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

# Enable VCD waveform
ifdef VCD
	VL_FLAGS += --trace --trace-structs
	CXXFLAGS += -DVCD_OUTPUT
endif

# Enable SAIF tracing
ifdef SAIF
	VL_FLAGS += --trace-saif
	CXXFLAGS += -DSAIF_OUTPUT
endif

all: $(DESTDIR)/$(PROJECT)

# verilator --build runs its own make (parallelized by -j in VL_FLAGS) that
# cannot reach this make's jobserver, so it gets a clean MAKEFLAGS.
$(DESTDIR)/$(PROJECT): $(SRCS) $(RTL_SRCS)
	MAKEFLAGS= $(VERILATOR_PATH)/bin/verilator --build $(VL_FLAGS) $(SRCS) -CFLAGS '$(CXXFLAGS)' -LDFLAGS '$(LDFLAGS)' -MMD -o ../$@

run: $(DESTDIR)/$(PROJECT)
	$(DESTDIR)/$(PROJECT) $(OPTS)

waves: trace.vcd
	gtkwave -o trace.vcd

clean:
	rm -rf *.vcd obj_dir $(DESTDIR)/$(PROJECT)