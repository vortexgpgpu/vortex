ROOT_DIR := $(realpath ../../../..)
include $(ROOT_DIR)/config.mk

DEVICE ?= xcu55c-fsvh2892-2L-e

MAX_JOBS ?= 8

VIVADO := $(XILINX_VIVADO)/bin/vivado

SRC_DIR := $(VORTEX_HOME)/hw/syn/xilinx/sandbox

RTL_DIR := $(VORTEX_HOME)/hw/rtl
DPI_DIR := $(VORTEX_HOME)/hw/dpi
AFU_DIR := $(RTL_DIR)/afu/xrt
SCRIPT_DIR := $(VORTEX_HOME)/hw/scripts

KERNEL ?= fibonacci

NCPUS := $(shell lscpu | grep "^Core(s) per socket:" | awk '{print $$4}')
JOBS ?= $(shell echo $$(( $(NCPUS) > $(MAX_JOBS) ? $(MAX_JOBS) : $(NCPUS) )))

# include paths
FPU_INCLUDE = -I$(RTL_DIR)/fpu
ifneq (,$(findstring FPU_FPNEW,$(CONFIGS)))
	FPU_INCLUDE += -J$(THIRD_PARTY_DIR)/cvfpu/src/common_cells/include -J$(THIRD_PARTY_DIR)/cvfpu/src/common_cells/src -J$(THIRD_PARTY_DIR)/cvfpu/src/fpu_div_sqrt_mvp/hdl -J$(THIRD_PARTY_DIR)/cvfpu/src
endif
RTL_INCLUDE = -I$(RTL_DIR) -I$(RTL_DIR)/libs -I$(RTL_DIR)/interfaces -I$(RTL_DIR)/core -I$(RTL_DIR)/mem -I$(RTL_DIR)/cache
RTL_INCLUDE += $(FPU_INCLUDE)
RTL_INCLUDE += -I$(SRC_DIR)

# compilation flags
CFLAGS += -DNDEBUG -DSYNTHESIS -DVIVADO
CFLAGS += $(CONFIGS)
CFLAGS += $(RTL_INCLUDE)
CFLAGS += -DEXT_F_DISABLE

# update memory layout for 2MB RAM
CFLAGS += -DSTARTUP_ADDR=32\'h80000
CFLAGS += -DSTACK_BASE_ADDR=32\'hFF000

all: build

$(KERNEL).bin:
	$(MAKE) -C $(ROOT_DIR)/kernel clean
	STACK_BASE_ADDR=0xFF000 $(MAKE) -C $(ROOT_DIR)/kernel
	$(MAKE) -C $(ROOT_DIR)/tests/kernel/$(KERNEL) clean
	STARTUP_ADDR=0x8000 $(MAKE) -C $(ROOT_DIR)/tests/kernel/$(KERNEL)
	cp $(ROOT_DIR)/tests/kernel/$(KERNEL)/$(KERNEL).bin $(KERNEL).bin

kernel.bin.coe: $(KERNEL).bin
	$(SCRIPT_DIR)/bin2coe.py --out=$@ --binfile=8192:$(KERNEL).bin --depth=16384 --wordsize=64 --little_endian

gen-sources: project_1/sources.txt
project_1/sources.txt:
	mkdir -p project_1
	$(SCRIPT_DIR)/gen_sources.sh $(CFLAGS) -P -Cproject_1/src -Oproject_1/sources.txt

build: done.dcp
done.dcp: project_1/sources.txt kernel.bin.coe project.tcl
	MAX_JOBS=$(JOBS) $(VIVADO) -mode batch -source project.tcl -tclargs $(DEVICE) project_1/sources.txt $(SCRIPT_DIR)
	echo done > done.dcp

run: project_1/project_1.xpr
	$(VIVADO) project_1/project_1.xpr &

clean:
	rm -rf project_1 project1.tcl $(KERNEL).bin kernel.bin.coe
	rm -rf .Xil *.log *.jou *.dcp *.rpt
