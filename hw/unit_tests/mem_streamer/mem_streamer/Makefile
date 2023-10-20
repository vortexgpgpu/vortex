#---------------------------------------------------------
# Makefile to compile and test the memory stream unit
#---------------------------------------------------------

TOP = VX_mem_scheduler
PARAMS += -GNUM_REQS=4 -GADDRW=8 -GDATAW=8 -GTAGW=8 -GWORD_SIZE=1 -GQUEUE_SIZE=4

ifdef RSP_PARTIAL
	PARAMS += -GRSP_PARTIAL=$(RSP_PARTIAL)
endif

ifdef DUPLICATE_ADDR
	PARAMS += -GDUPLICATE_ADDR=$(DUPLICATE_ADDR)
endif

RTL_DIR = ../../../rtl
DPI_DIR = ../../../dpi

RTL_INCLUDE = -I$(RTL_DIR) -I$(RTL_DIR)/libs -I$(DPI_DIR)

VERILATOR_ROOT = /opt/verilator
VERILATOR ?= $(VERILATOR_ROOT)/bin/verilator

VL_FLAGS += --exe --cc $(TOP).sv --top-module $(TOP)
VL_FLAGS += --language 1800-2009 --assert -Wall
VL_FLAGS += -Wno-DECLFILENAME -Wno-REDEFMACRO
VL_FLAGS += --x-initial unique --x-assign unique
VL_FLAGS += --trace
VL_FLAGS += $(RTL_INCLUDE)
VL_FLAGS += $(PARAMS)

SRCS += memsim.cpp ram.cpp $(DPI_DIR)/util_dpi.cpp

CXXFLAGS += -std=c++11 -Wall -Wextra -Wfatal-errors -Wno-array-bounds -Wno-maybe-uninitialized
CXXFLAGS += -I../../../../dpi -I../../../../

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

waves: trace.vcd
	@echo 
	@echo "### TRACE ###"
	gtkwave -o trace.vcd

clean:
	@echo
	@echo "### CLEAN ###"
	-rm -rf obj_dir *.vcd *.log

#---------------------------------------------------------

