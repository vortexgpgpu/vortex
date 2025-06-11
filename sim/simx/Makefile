include ../common.mk

DESTDIR ?= $(CURDIR)
OBJ_DIR = $(DESTDIR)/obj
CONFIG_FILE = $(OBJ_DIR)/config_flags

SRC_DIR = $(VORTEX_HOME)/sim/simx

CXXFLAGS += -std=c++17 -Wall -Wextra -Wfatal-errors
CXXFLAGS += -fPIC -Wno-maybe-uninitialized
CXXFLAGS += -I$(SRC_DIR) -I$(COMMON_DIR) -I$(ROOT_DIR)/hw
CXXFLAGS += -I$(THIRD_PARTY_DIR)/softfloat/source/include
CXXFLAGS += -I$(THIRD_PARTY_DIR)/ramulator/ext/spdlog/include
CXXFLAGS += -I$(THIRD_PARTY_DIR)/ramulator/ext/yaml-cpp/include
CXXFLAGS += -I$(THIRD_PARTY_DIR)/ramulator/src
CXXFLAGS += -DXLEN_$(XLEN)
CXXFLAGS += $(CONFIGS)

LDFLAGS += $(THIRD_PARTY_DIR)/softfloat/build/Linux-x86_64-GCC/softfloat.a
LDFLAGS += -Wl,-rpath,$(THIRD_PARTY_DIR)/ramulator -L$(THIRD_PARTY_DIR)/ramulator -lramulator

# Source files definition
SRCS = $(COMMON_DIR)/util.cpp $(COMMON_DIR)/mem.cpp $(COMMON_DIR)/softfloat_ext.cpp $(COMMON_DIR)/rvfloats.cpp $(COMMON_DIR)/dram_sim.cpp
SRCS += $(SRC_DIR)/processor.cpp $(SRC_DIR)/cluster.cpp $(SRC_DIR)/socket.cpp $(SRC_DIR)/core.cpp $(SRC_DIR)/emulator.cpp
SRCS += $(SRC_DIR)/decode.cpp $(SRC_DIR)/opc_unit.cpp $(SRC_DIR)/dispatcher.cpp
SRCS += $(SRC_DIR)/execute.cpp $(SRC_DIR)/func_unit.cpp
SRCS += $(SRC_DIR)/cache_sim.cpp $(SRC_DIR)/mem_sim.cpp $(SRC_DIR)/local_mem.cpp $(SRC_DIR)/mem_coalescer.cpp
SRCS += $(SRC_DIR)/dcrs.cpp $(SRC_DIR)/types.cpp

# Add V extension sources
ifneq ($(findstring -DEXT_V_ENABLE, $(CONFIGS)),)
  	SRCS += $(SRC_DIR)/voperands.cpp
  	SRCS += $(SRC_DIR)/vopc_unit.cpp
  	SRCS += $(SRC_DIR)/vec_unit.cpp
else
	SRCS += $(SRC_DIR)/operands.cpp
endif
# Add TPU extension sources
ifneq ($(findstring -DEXT_TPU_ENABLE, $(CONFIGS)),)
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
COMMON_SRCS := $(filter $(COMMON_DIR)/%.cpp,$(SRCS))
SRC_SRCS    := $(filter $(SRC_DIR)/%.cpp,$(SRCS))
COMMON_OBJS := $(patsubst $(COMMON_DIR)/%.cpp,$(OBJ_DIR)/common/%.o,$(COMMON_SRCS))
SRC_OBJS    := $(patsubst $(SRC_DIR)/%.cpp,$(OBJ_DIR)/%.o,$(SRC_SRCS))
OBJS        := $(COMMON_OBJS) $(SRC_OBJS)
MAIN_OBJ    := $(OBJ_DIR)/main.o

DEPS := $(OBJS:.o=.d) $(MAIN_OBJ:.o=.d)

# generate .d files alongside .o files
CXXFLAGS += -MMD -MP -MF $(@:.o=.d)

# optional: pipe through ccache if you have it
CXX := $(if $(shell which ccache),ccache $(CXX),$(CXX))

PROJECT := simx

.PHONY: all clean force

all: $(DESTDIR)/$(PROJECT)

# build common object files
$(OBJ_DIR)/common/%.o: $(COMMON_DIR)/%.cpp
	@mkdir -p $(@D)
	$(CXX) $(CXXFLAGS) -c $< -o $@

# build source object files
$(OBJ_DIR)/%.o: $(SRC_DIR)/%.cpp $(CONFIG_FILE)
	@mkdir -p $(@D)
	$(CXX) $(CXXFLAGS) -c $< -o $@

# build main object file
$(MAIN_OBJ): $(SRC_DIR)/main.cpp $(CONFIG_FILE)
	@mkdir -p $(@D)
	$(CXX) $(CXXFLAGS) -DSTARTUP_ADDR=0x80000000 -c $< -o $@

# Config flags tracking
$(CONFIG_FILE): force
	@mkdir -p $(@D)
	@test -f $@ && cp $@ $@.old || true
	@echo "$(CXXFLAGS)" > $@.tmp
	@diff -q $@.tmp $@.old 2>/dev/null || cp $@.tmp $@
	@rm -f $@.tmp $@.old

# Main executable
$(DESTDIR)/$(PROJECT): $(OBJS) $(MAIN_OBJ)
	$(CXX) $(CXXFLAGS) $^ $(LDFLAGS) -o $@

# Shared library
$(DESTDIR)/lib$(PROJECT).so: $(OBJS)
	$(CXX) $(CXXFLAGS) $^ -shared $(LDFLAGS) -o $@

# include the auto-generated header deps; silences warnings if missing
-include $(DEPS)

clean-lib:
	rm -f $(DESTDIR)/lib$(PROJECT).so

clean-exe:
	rm -f $(DESTDIR)/$(PROJECT)

clean-obj:
	rm -rf $(OBJ_DIR)

clean: clean-lib clean-exe clean-obj