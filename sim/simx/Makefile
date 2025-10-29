include ../common.mk

DESTDIR ?= $(CURDIR)

OBJ_DIR = $(DESTDIR)/obj
CONFIG_FILE = $(DESTDIR)/simx_config.stamp
SRC_DIR = $(VORTEX_HOME)/sim/simx

CXXFLAGS += -std=c++17 -Wall -Wextra -Wfatal-errors
CXXFLAGS += -fPIC -Wno-maybe-uninitialized
CXXFLAGS += -I$(SRC_DIR) -I$(SW_COMMON_DIR) -I$(ROOT_DIR)/hw
CXXFLAGS += -I$(THIRD_PARTY_DIR)/softfloat/source/include
CXXFLAGS += -I$(THIRD_PARTY_DIR)/ramulator/ext/spdlog/include
CXXFLAGS += -I$(THIRD_PARTY_DIR)/ramulator/ext/yaml-cpp/include
CXXFLAGS += -I$(THIRD_PARTY_DIR)/ramulator/src
CXXFLAGS += -DXLEN_$(XLEN)
CXXFLAGS += $(CONFIGS)

LDFLAGS += $(THIRD_PARTY_DIR)/softfloat/build/Linux-x86_64-GCC/softfloat.a
LDFLAGS += -Wl,-rpath,$(THIRD_PARTY_DIR)/ramulator -L$(THIRD_PARTY_DIR)/ramulator -lramulator

# Source files definition
SRCS = $(SW_COMMON_DIR)/util.cpp $(SW_COMMON_DIR)/mem.cpp $(SW_COMMON_DIR)/softfloat_ext.cpp $(SW_COMMON_DIR)/rvfloats.cpp $(SW_COMMON_DIR)/dram_sim.cpp
SRCS += $(SRC_DIR)/processor.cpp $(SRC_DIR)/cluster.cpp $(SRC_DIR)/socket.cpp $(SRC_DIR)/core.cpp $(SRC_DIR)/emulator.cpp
SRCS += $(SRC_DIR)/decode.cpp $(SRC_DIR)/opc_unit.cpp $(SRC_DIR)/dispatcher.cpp
SRCS += $(SRC_DIR)/execute.cpp $(SRC_DIR)/func_unit.cpp
SRCS += $(SRC_DIR)/cache_sim.cpp $(SRC_DIR)/mem_sim.cpp $(SRC_DIR)/local_mem.cpp $(SRC_DIR)/mem_coalescer.cpp
SRCS += $(SRC_DIR)/dcrs.cpp $(SRC_DIR)/types.cpp

VORTEX_SST_SRCS := $(SRC_DIR)/vortex_simulator.cpp
VORTEX_SST_SRCS += $(SRC_DIR)/VortexGPGPU.cpp

# SST flags
SST_CFLAGS := $(shell pkg-config --cflags SST-14.1)
SST_CFLAGS += -I../../../sst/sst/sst-core/include/
SST_LFLAGS := $(shell pkg-config --libs SST-14.1)

# Add V extension sources
ifneq ($(findstring -DEXT_V_ENABLE, $(CONFIGS)),)
  	SRCS += $(SRC_DIR)/voperands.cpp
  	SRCS += $(SRC_DIR)/vopc_unit.cpp
  	SRCS += $(SRC_DIR)/vec_unit.cpp
else
	SRCS += $(SRC_DIR)/operands.cpp
endif
# Add TCU extension sources
ifneq ($(findstring -DEXT_TCU_ENABLE, $(CONFIGS)),)
  	SRCS += $(SRC_DIR)/tensor_unit.cpp
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
SRC_SRCS    := $(filter $(SRC_DIR)/%.cpp,$(SRCS))
COMMON_OBJS := $(patsubst $(SW_COMMON_DIR)/%.cpp,$(OBJ_DIR)/common/%.o,$(COMMON_SRCS))
SRC_OBJS    := $(patsubst $(SRC_DIR)/%.cpp,$(OBJ_DIR)/%.o,$(SRC_SRCS))
VORTEX_SST_OBJS := $(patsubst $(SRC_DIR)/%.cpp,$(OBJ_DIR)/%.o,$(VORTEX_SST_SRCS))
OBJS        := $(COMMON_OBJS) $(SRC_OBJS)
MAIN_OBJ    := $(OBJ_DIR)/main.o

DEPS := $(OBJS:.o=.d) $(MAIN_OBJ:.o=.d) $(VORTEX_SST_OBJS:.o=.d)

# optional: pipe through ccache if you have it
CXX := $(if $(shell which ccache),ccache $(CXX),$(CXX))

PROJECT := simx
VORTEX_LIB := libvortex.so

.PHONY: all force clean clean-lib clean-exe clean-obj libvortex clean-libvortex

#ifdef USE_SST
all: $(DESTDIR)/$(PROJECT) $(DESTDIR)/$(VORTEX_LIB)
#else
#all: $(DESTDIR)/$(PROJECT)
#endif

# build common object files
$(OBJ_DIR)/common/%.o: $(SW_COMMON_DIR)/%.cpp $(CONFIG_FILE)
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