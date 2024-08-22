ROOT_DIR := $(realpath ../../../..)
include $(ROOT_DIR)/config.mk

DEVICE ?= xcu55c-fsvh2892-2L-e

VIVADO := $(XILINX_VIVADO)/bin/vivado

SRC_DIR := $(VORTEX_HOME)/hw/syn/xilinx/sandbox

RTL_DIR := $(VORTEX_HOME)/hw/rtl
DPI_DIR := $(VORTEX_HOME)/hw/dpi
AFU_DIR := $(RTL_DIR)/afu/xrt
SCRIPT_DIR := $(VORTEX_HOME)/hw/scripts

KERNEL ?= fibonacci

COE_FILE := $(shell realpath kernel.bin.coe)
ESCAPED_COE_FILE := $(shell echo "$(COE_FILE)" | sed -e 's/[\/&]/\\&/g')

# include paths
FPU_INCLUDE = -I$(RTL_DIR)/fpu
ifneq (,$(findstring FPU_FPNEW,$(CONFIGS)))
	FPU_INCLUDE += -I$(THIRD_PARTY_DIR)/fpnew/src/common_cells/include -I$(THIRD_PARTY_DIR)/fpnew/src/common_cells/src -I$(THIRD_PARTY_DIR)/fpnew/src/fpu_div_sqrt_mvp/hdl -I$(THIRD_PARTY_DIR)/fpnew/src
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

project2.tcl: project.tcl
	@sed -e "s/@COE_FILE@/$(ESCAPED_COE_FILE)/g" $< > $@

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

build: project_1/project_1.xpr
project_1/project_1.xpr: project_1/sources.txt kernel.bin.coe project2.tcl
	$(VIVADO) -mode batch -source project2.tcl -tclargs $(DEVICE) project_1/sources.txt $(SCRIPT_DIR)

run: project_1/project_1.xpr
	$(VIVADO) project_1/project_1.xpr &

clean:
	rm -rf project_1 project2.tcl $(KERNEL).bin kernel.bin.coe
	rm -rf .Xil *.log *.jou
