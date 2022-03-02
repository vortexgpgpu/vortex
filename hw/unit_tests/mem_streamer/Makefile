#---------------------------------------------------------
# Makefile to compile and test the memory stream unit
#---------------------------------------------------------

TOP = VX_mem_streamer_test

RTL_DIR = ../../rtl
DPI_DIR = ../../dpi

RTL_INCLUDE = -I$(RTL_DIR) -I$(RTL_DIR)/libs -I$(DPI_DIR)

VERILATOR ?= $(VERILATOR_ROOT)/bin/verilator

VL_FLAGS += --exe --cc $(TOP).sv --top-module $(TOP)
VL_FLAGS += --language 1800-2009 --assert -Wall
VL_FLAGS += -Wno-DECLFILENAME -Wno-REDEFMACRO
VL_FLAGS += --x-initial unique --x-assign unique
VL_FLAGS += --trace
VL_FLAGS += $(RTL_INCLUDE)

SRCS += memsim.cpp $(DPI_DIR)/util_dpi.cpp

CXXFLAGS += -std=c++11 -Wall -Wextra -Wfatal-errors -Wno-array-bounds -Wno-maybe-uninitialized
CXXFLAGS += ../../../dpi

default: run

gen: $(SRCS)
	@echo 
	@echo "### VERILATE ###"
	$(VERILATOR) $(VL_FLAGS) $^ -CFLAGS '$(CXXFLAGS)'

build: gen
	@echo 
	@echo "### BUILD ###"
	$(MAKE) -C obj_dir -j 4 -f V$(TOP).mk

run: build
	@echo
	@echo "### RUN ###"
	obj_dir/V$(TOP)

clean:
	@echo
	@echo "### CLEAN ###"
	-rm -rf obj_dir *.vcd

#---------------------------------------------------------

