### Directories 
# Vortex
RTL_DIR := rtl
AFU_DIR := $(RTL_DIR)/afu
CACHE_DIR := $(RTL_DIR)/cache
FP_CORE_DIR := $(RTL_DIR)/fp_cores
INTERFACES_DIR := $(RTL_DIR)/interfaces
LIBS_DIR := $(RTL_DIR)/libs
TEX_UNIT_DIR := $(RTL_DIR)/tex_unit
DPI_DIR := dpi

# FPNEW 
FPNEW_DIR := ../third_party/fpnew/src

# AHB
AHB_DIR := $(RTL_DIR)/ahb
AHB_INTF_DIR := $(AHB_DIR)/ahb_if
AHB_INTF_FILES := $(wildcard $(AHB_INTF_DIR)/*.sv)
AHB_RTL_FILES := $(AHB_DIR)/ahb_manager/src/ahb_manager.sv
AHB_TB := tb/VX_ahb_tb.sv

INCL := 
INCL := include

#Interface files 
INTF_FILES := 
INTF_FILES += $(wildcard $(INTERFACES_DIR)/*.sv)

#FPNEW files 
FPNEW_FILES := 
FPNEW_FILES := $(wildcard $(FPNEW_DIR)**/*.sv) 

#FPNEW INC
FPNEW_INC := $(FPNEW_DIR)

#Design files 
RTL_FILES := 
#RTL_FILES += $(wildcard $(AFU_DIR)/*.sv)
RTL_FILES += $(wildcard $(CACHE_DIR)/*.sv)
RTL_FILES += $(wildcard $(FP_CORE_DIR)/*.sv)
RTL_FILES += $(wildcard $(LIBS_DIR)/*.sv)
RTL_FILES += $(wildcard $(TEXT_UNIT_DIR)/*.sv)

RTL_FILES += $(wildcard $(RTL_DIR)/*.sv)



# RTL_FILES := $(wildcard $(AFU_DIR)/*.vh)
# RTL_FILES := $(wildcard $(CACHE_DIR)/*.vh)
# RTL_FILES := $(wildcard $(FP_CORE_DIR)/*.vh)

WORKDIR := work

# Compiler/simulator options
VSIMOPTS := -lib $(WORKDIR)
VLOGOPTS := -sv -work $(WORKDIR) -incr -vopt 

all: $(WORKDIR) compile run 

compile: $(WORKDIR)
		@vlog $(VLOGOPTS) +incdir+$(INCL)+dpi $(INTF_FILES) 
		@vlog $(VLOGOPTS) +incdir+$(INCL) $(AHB_INTF_FILES)
		@vlog $(VLOGOPTS) +incdir+$(INCL) $(AHB_RTL_FILES)
		@vlog $(VLOGOPTS) $(FPNEW_DIR)/fpnew_pkg.sv
		@vlog $(VLOGOPTS) $(FPNEW_FILES)
		@vlog $(VLOGOPTS) +incdir+$(INCL)+dpi $(RTL_FILES)
		@vlog $(VLOGOPTS) +incdir+$(INCL) $(AHB_TB)

run:
		vsim $(VSIMOPTS) -c \
		VX_ahb_tb \
		-coverage \
		-do "run -all" 

$(WORKDIR):
	@vlib $(WORKDIR)

clean: 
	rm -rf work $(WORKDIR) transcript *.vstf 
	rm *.wlf


# RTL_DIR=./rtl
# SCRIPT_DIR=./scripts

# all: VX_config.h

# VX_config.h: $(RTL_DIR)/VX_config.vh	 
# 	$(SCRIPT_DIR)/gen_config.py -i $(RTL_DIR)/VX_config.vh -o VX_config.h

# clean:
# 	rm -f VX_config.h

# .PHONY: VX_config.h