include ../common.mk

DESTDIR ?= $(CURDIR)
USE_SST ?= 0
#SST_PKG ?= SST-14.1 # default SST package name

OBJ_DIR = $(DESTDIR)/obj
CONFIG_FILE = $(DESTDIR)/simx_config.stamp
SRC_DIR = $(VORTEX_HOME)/sim/simx

CXXFLAGS += -std=c++17 -Wall -Wextra -Wfatal-errors -Werror
CXXFLAGS += -fPIC -Wno-maybe-uninitialized
CXXFLAGS += -I$(SRC_DIR) -I$(SRC_DIR)/dxa -I$(SRC_DIR)/kmu -I$(SRC_DIR)/mem -I$(SW_COMMON_DIR) -I$(SIM_COMMON_DIR) -I$(ROOT_DIR)/sw -I$(ROOT_DIR)/hw
CXXFLAGS += -I$(THIRD_PARTY_DIR)/softfloat/source/include
CXXFLAGS += -I$(THIRD_PARTY_DIR)/ramulator/ext/spdlog/include
CXXFLAGS += -I$(THIRD_PARTY_DIR)/ramulator/ext/yaml-cpp/include
CXXFLAGS += -I$(THIRD_PARTY_DIR)/ramulator/src
CXXFLAGS += -DXLEN_$(XLEN)
CXXFLAGS += $(CONFIGS)

LDFLAGS += $(THIRD_PARTY_DIR)/softfloat/build/Linux-x86_64-GCC/softfloat.a
LDFLAGS += -Wl,-rpath,$(THIRD_PARTY_DIR)/ramulator -L$(THIRD_PARTY_DIR)/ramulator -lramulator

XCONFIGS := $(shell python3 $(ROOT_DIR)/ci/gen_config.py --config=$(VORTEX_HOME)/VX_config.toml --cflags='$(CONFIGS)')

# Source files definition
SRCS = $(SW_COMMON_DIR)/util.cpp $(SIM_COMMON_DIR)/mem.cpp $(SW_COMMON_DIR)/softfloat_ext.cpp $(SW_COMMON_DIR)/rvfloats.cpp $(SIM_COMMON_DIR)/dram_sim.cpp
SRCS += $(SRC_DIR)/processor.cpp $(SRC_DIR)/cluster.cpp $(SRC_DIR)/socket.cpp $(SRC_DIR)/core.cpp $(SRC_DIR)/scheduler.cpp $(SRC_DIR)/cta_dispatcher.cpp $(SRC_DIR)/barrier_unit.cpp
SRCS += $(SRC_DIR)/kmu/kmu.cpp
SRCS += $(SRC_DIR)/decode.cpp $(SRC_DIR)/scoreboard.cpp $(SRC_DIR)/sequencer.cpp $(SRC_DIR)/opc_unit.cpp $(SRC_DIR)/dispatcher.cpp
SRCS += $(SRC_DIR)/alu_unit.cpp $(SRC_DIR)/fpu_unit.cpp $(SRC_DIR)/lsu_unit.cpp $(SRC_DIR)/sfu_unit.cpp $(SRC_DIR)/csr_unit.cpp
SRCS += $(SRC_DIR)/mem/cache.cpp $(SRC_DIR)/mem/cache_cluster.cpp $(SRC_DIR)/mem/memory.cpp $(SRC_DIR)/mem/local_mem.cpp $(SRC_DIR)/mem/mem_coalescer.cpp
SRCS += $(SRC_DIR)/mem/local_mem_switch.cpp $(SRC_DIR)/mem/lsu_mem_adapter.cpp

# Add V extension sources
ifneq ($(filter -DEXT_V_ENABLE, $(XCONFIGS)),)
	CXXFLAGS += -I$(SRC_DIR)/vpu
	SRCS += $(SRC_DIR)/vpu/voperands.cpp
	SRCS += $(SRC_DIR)/vpu/vopc_unit.cpp
	SRCS += $(SRC_DIR)/vpu/vec_unit.cpp
else
	SRCS += $(SRC_DIR)/operands.cpp
endif
# Add TCU extension sources
ifneq ($(filter -DEXT_TCU_ENABLE, $(XCONFIGS)),)
	CXXFLAGS += -I$(SRC_DIR)/tcu
	SRCS += $(SRC_DIR)/tcu/tcu_unit.cpp
	SRCS += $(SRC_DIR)/tcu/tcu_tbuf.cpp
endif

# Add DXA extension sources
ifneq ($(filter -DEXT_DXA_ENABLE, $(XCONFIGS)),)
	SRCS += $(SRC_DIR)/dxa/dxa_core.cpp
endif

# Debugging
ifdef DEBUG
	CXXFLAGS += -g -O0 -DDEBUG_LEVEL=$(DEBUG)
else
	CXXFLAGS += -O2 -DNDEBUG
endif

# Enable perf counters
ifdef PERF
	CXXFLAGS += -DPERF_ENABLE
endif

# Convert sources to object files (in build directory)
COMMON_SRCS := $(filter $(SW_COMMON_DIR)/%.cpp,$(SRCS))
SIM_COMMON_SRCS := $(filter $(SIM_COMMON_DIR)/%.cpp,$(SRCS))
SRC_SRCS    := $(filter $(SRC_DIR)/%.cpp,$(SRCS))
COMMON_OBJS := $(patsubst $(SW_COMMON_DIR)/%.cpp,$(OBJ_DIR)/common/%.o,$(COMMON_SRCS))
SIM_COMMON_OBJS := $(patsubst $(SIM_COMMON_DIR)/%.cpp,$(OBJ_DIR)/sim_common/%.o,$(SIM_COMMON_SRCS))
SRC_OBJS    := $(patsubst $(SRC_DIR)/%.cpp,$(OBJ_DIR)/%.o,$(SRC_SRCS))
OBJS            := $(COMMON_OBJS) $(SIM_COMMON_OBJS) $(SRC_OBJS)
MAIN_OBJ    := $(OBJ_DIR)/main.o

DEPS := $(OBJS:.o=.d) $(MAIN_OBJ:.o=.d)

# generate .d files alongside .o files
CXXFLAGS += -MMD -MP -MF $(@:.o=.d)

ifeq ($(USE_SST), 1)
VORTEX_SST_OBJS := $(patsubst $(SRC_DIR)/%.cpp,$(OBJ_DIR)/%.o,$(VORTEX_SST_SRCS))
DEPS += $(VORTEX_SST_OBJS:.o=.d)
endif


# optional: pipe through ccache if you have it
CXX := $(if $(shell which ccache),ccache $(CXX),$(CXX))

PROJECT := simx
VORTEX_LIB := libvortex.so

.PHONY: all force clean clean-lib clean-exe clean-obj libvortex clean-libvortex

ifeq ($(USE_SST), 1)
all: $(DESTDIR)/$(PROJECT) $(DESTDIR)/$(VORTEX_LIB)
else
all: $(DESTDIR)/$(PROJECT)
endif

# build common object files
$(OBJ_DIR)/common/%.o: $(SW_COMMON_DIR)/%.cpp $(CONFIG_FILE)
	@mkdir -p $(@D)
	$(CXX) $(CXXFLAGS) -c $< -o $@

# build sim-only common object files
$(OBJ_DIR)/sim_common/%.o: $(SIM_COMMON_DIR)/%.cpp $(CONFIG_FILE)
	@mkdir -p $(@D)
	$(CXX) $(CXXFLAGS) -c $< -o $@

# build source object files
$(OBJ_DIR)/%.o: $(SRC_DIR)/%.cpp $(CONFIG_FILE)
	@mkdir -p $(@D)
	$(CXX) $(CXXFLAGS) -c $< -o $@

# build SST-specific source object files
$(VORTEX_SST_OBJS): $(OBJ_DIR)/%.o: $(SRC_DIR)/%.cpp $(CONFIG_FILE)
	@mkdir -p $(@D)
	$(CXX) $(CXXFLAGS) $(SST_CFLAGS) -c $< -o $@

# build main object file
$(MAIN_OBJ): $(SRC_DIR)/main.cpp $(CONFIG_FILE)
	@mkdir -p $(@D)
	$(CXX) $(CXXFLAGS) -DSTARTUP_ADDR=0x80000000 -c $< -o $@

# Main executable
$(DESTDIR)/$(PROJECT): $(OBJS) $(MAIN_OBJ)
	$(CXX) $(CXXFLAGS) $^ $(LDFLAGS) -o $@

# Shared library
$(DESTDIR)/lib$(PROJECT).so: $(OBJS)
	$(CXX) $(CXXFLAGS) $^ -shared $(LDFLAGS) -o $@

# Vortex SST simulator component shared library
libvortex: $(DESTDIR)/$(VORTEX_LIB)

$(DESTDIR)/$(VORTEX_LIB): $(OBJS) $(VORTEX_SST_OBJS)
	$(CXX) $(CXXFLAGS) $(SST_CFLAGS) \
	-I./sim/simx \
	$(OBJS) $(VORTEX_SST_SRCS) \
	-shared -o $@ \
	$(LDFLAGS) $(SST_LFLAGS)

# updates the timestamp when flags changed.
$(CONFIG_FILE): force
	@mkdir -p $(@D)
	@printf '%s\n' "$(CXXFLAGS)" > $@.tmp
	@if ! cmp -s $@.tmp $@; then \
	  mv $@.tmp $@; \
	else \
	  rm $@.tmp; \
	fi

# include the auto-generated header deps; silences warnings if missing
-include $(DEPS)

clean-lib:
	rm -f $(DESTDIR)/lib$(PROJECT).so

clean-libvortex:
	rm -f $(DESTDIR)/libvortex.so

clean-exe:
	rm -f $(DESTDIR)/$(PROJECT)

clean-obj:
	rm -rf $(OBJ_DIR)

clean: clean-lib clean-exe clean-obj clean-libvortex