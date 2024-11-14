ROOT_DIR := $(realpath ../../../../../..)
include $(ROOT_DIR)/config.mk

DEVICE ?= xcu55c-fsvh2892-2L-e

MAX_JOBS ?= 8

VIVADO := $(XILINX_VIVADO)/bin/vivado

SRC_DIR := $(VORTEX_HOME)/hw/syn/xilinx/dut

RTL_DIR := $(VORTEX_HOME)/hw/rtl
AFU_DIR := $(RTL_DIR)/afu/xrt
SCRIPT_DIR := $(VORTEX_HOME)/hw/scripts

NCPUS := $(shell lscpu | grep "^Core(s) per socket:" | awk '{print $$4}')
JOBS ?= $(shell echo $$(( $(NCPUS) > $(MAX_JOBS) ? $(MAX_JOBS) : $(NCPUS) )))

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
ifdef FPU_IP
	MAX_JOBS=$(JOBS) FPU_IP=project_1/ip SCRIPT_DIR=$(SCRIPT_DIR) $(VIVADO) -mode batch -source $(SRC_DIR)/project.tcl -tclargs $(TOP_LEVEL_ENTITY) $(DEVICE) project_1/sources.txt $(SRC_DIR)/project.xdc
else
	MAX_JOBS=$(JOBS) SCRIPT_DIR=$(SCRIPT_DIR) $(VIVADO) -mode batch -source $(SRC_DIR)/project.tcl -tclargs $(TOP_LEVEL_ENTITY) $(DEVICE) project_1/sources.txt $(SRC_DIR)/project.xdc
endif

clean:
	rm -rf project_1
	rm -rf .Xil
	rm -f *.rpt
	rm -f vivado*.log
	rm -f vivado*.jou

.PHONY: all gen-sources build clean