ROOT_DIR := $(realpath ../../../../../..)
include $(ROOT_DIR)/config.mk

DEVICE ?= xcu55c-fsvh2892-2L-e

VIVADO := $(XILINX_VIVADO)/bin/vivado

SRC_DIR := $(VORTEX_HOME)/hw/syn/xilinx/dut

RTL_DIR := $(VORTEX_HOME)/hw/rtl
AFU_DIR := $(RTL_DIR)/afu/xrt
SCRIPT_DIR := $(VORTEX_HOME)/hw/scripts

CONFIGS += -DNDEBUG
CONFIGS += -DVIVADO
CONFIGS += -DSYNTHESIS

# Build targets
all: $(PROJECT).xpr

gen-sources: project_1/sources.txt
project_1/sources.txt:
	mkdir -p project_1
	$(SCRIPT_DIR)/gen_sources.sh $(CONFIGS) $(RTL_INCLUDE) -T$(TOP_LEVEL_ENTITY) -P -Cproject_1/src -Oproject_1/sources.txt

build: $(PROJECT).xpr
$(PROJECT).xpr: project_1/sources.txt
	$(VIVADO) -mode batch -source $(SRC_DIR)/project.tcl -tclargs $(TOP_LEVEL_ENTITY) $(DEVICE) project_1/sources.txt $(SRC_DIR)/project.xdc $(SCRIPT_DIR)

clean:
	rm -rf project_1
	rm -rf .Xil
	rm -f *.rpt
	rm -f vivado*.log
	rm -f vivado*.jou

.PHONY: all gen-sources build clean