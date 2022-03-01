#---------------------------------------------------------
# Makefile to compile and test the memory stream unit
#---------------------------------------------------------

TOP = VX_mem_streamer

HW_DIR = ../..
RTL_DIR = $(HW_DIR)/rtl
DPI_DIR = $(HW_DIR)/dpi
INCLUDE = -I$(RTL_DIR) -I$(DPI_DIR) -I$(RTL_DIR)/libs
SRCS = memsim.cpp $(DPI_DIR)/util_dpi.cpp

VERILATOR ?= $(VERILATOR_ROOT)/bin/verilator

VL_FLAGS = 
# Generate C++ in executable form
VL_FLAGS += -exe -cc $(TOP).sv --top-module $(TOP)
# Optimize. Convert all x's to 0's
VL_FLAGS += -O2 -x-assign 0
# Warn about lint issues. May not want this on less solid designs
VL_FLAGS += -Wall -Wpedantic -Wno-DECLFILENAME -Wno-REDEFMACRO
# Generate trace
VL_FLAGS += --trace
# Include paths
VL_FLAGS += $(INCLUDE)

# CXXFLAGS += -std=c++11 -Wall -Wextra -Wfatal-errors -Wno-array-bounds
# CXXFLAGS += -fPIC -Wno-maybe-uninitialized
# CXXFLAGS += -I$(HW_DIR)

CF += -std=c++11 -fms-extensions -I../..

default: build

gen: $(SRCS)
	@echo "--VERILATE--"
	$(VERILATOR) $(VL_FLAGS) $^ -CFLAGS '$(CF)'

build: gen
	@echo "--COMPILE--"
	$(MAKE) -C obj_dir -j 4 -f V$(TOP).mk

run: build
	@echo "--RUN--"
	obj_dir/V$(TOP)

clean:
	@echo "--CLEAN--"
	-rm -rf obj_dir *.vcd

#---------------------------------------------------------

