.PHONY: help

help::
	$(ECHO) "Makefile Usage:"
	$(ECHO) "  make all TARGET=<sw_emu/hw_emu/hw> PLATFORM=<FPGA platform>"
	$(ECHO) "      Command to generate the design for specified Target and Device."
	$(ECHO) ""
	$(ECHO) "  make clean "
	$(ECHO) "      Command to remove the generated non-hardware files."
	$(ECHO) ""
	$(ECHO) "  make cleanall"
	$(ECHO) "      Command to remove all the generated files."
	$(ECHO) ""
	$(ECHO) "  make check TARGET=<sw_emu/hw_emu/hw> PLATFORM=<FPGA platform>"
	$(ECHO) "      Command to run application in emulation."
	$(ECHO) ""
	$(ECHO) "  make run_nimbix PLATFORM=<FPGA platform>"
	$(ECHO) "      Command to run application on Nimbix Cloud."
	$(ECHO) ""
	$(ECHO) "  make aws_build PLATFORM=<FPGA platform>"
	$(ECHO) "      Command to build AWS xclbin application on AWS Cloud."
	$(ECHO) ""

TARGET := hw
PLATFORM :=
XCLBIN := ./xclbin

VPP := $(XILINX_VITIS)/bin/v++

CP = cp -rf
RMDIR = rm -rf

ECHO:= @echo

XO_CONTAINER = $(XCLBIN)/vortex_afu.$(TARGET).xo

XCLBIN_CONTAINER = $(XCLBIN)/vortex_afu.$(TARGET).xclbin

# Kernel compiler global settings
CLFLAGS += -t $(TARGET) --platform $(PLATFORM) --save-temps

##########################################################################
# The below commands generate a XO file from a pre-exsisitng RTL kernel.
###########################################################################

VIVADO := $(XILINX_VIVADO)/bin/vivado
$(XO_CONTAINER): ./scripts/package_kernel.tcl ./scripts/gen_xo.tcl
	mkdir -p $(XCLBIN)
	$(VIVADO) -mode batch -source scripts/gen_xo.tcl -tclargs $(XO_CONTAINER) vortex_afu $(TARGET) $(PLATFORM)

###########################################################################
#END OF GENERATION OF XO
##########################################################################

.PHONY: all clean cleanall emconfig

all: check-devices $(XCLBIN_CONTAINER) emconfig

# Building kernel
$(XCLBIN_CONTAINER): $(XO_CONTAINER)
	mkdir -p $(XCLBIN)
	$(VPP) $(CLFLAGS) $(LDCLFLAGS) -lo $(XCLBIN_CONTAINER) $(XO_CONTAINER)

emconfig: $(XCLBIN)/emconfig.json
$(XCLBIN)/emconfig.json:
	emconfigutil --platform $(PLATFORM) --od $(XCLBIN)

# Cleaning stuff
clean:
	-$(RMDIR) $(XCLBIN)/{*sw_emu*,*hw_emu*}
	-$(RMDIR) TempConfig system_estimate.xtxt *.rpt
	-$(RMDIR) src/*.ll _v++_* .Xil emconfig.json dltmp* xmltmp* *.log *.jou

cleanall: clean
	-$(RMDIR) $(XCLBIN)
	-$(RMDIR) _x.*
	-$(RMDIR) ./tmp_kernel_pack* ./packaged_kernel* _x/

#######################################################################
# RTL Kernel only supports Hardware and Hardware Emulation.
# THis line is to check that
#########################################################################
ifneq ($(TARGET),$(findstring $(TARGET), hw hw_emu))
$(warning WARNING:Application supports only hw hw_emu TARGET. Please use the target for running the application)
endif

###################################################################
#check the devices avaiable
########################################################################

check-devices:
ifndef PLATFORM
	$(error PLATFORM not set. Please set the PLATFORM properly and rerun. Run "make help" for more details.)
endif

############################################################################
# check the VITIS environment
#############################################################################

ifndef XILINX_VITIS
$(error XILINX_VITIS variable is not set, please set correctly and rerun)
endif


#################################################################
# Enable profiling if needed
#####################################################################a

REPORT := no
PROFILE := no
DEBUG := no

#'estimate' for estimate report generation
#'system' for system report generation
ifneq ($(REPORT), no)
CLFLAGS += --report estimate
CLLDFLAGS += --report system
endif

#Generates profile summary report
ifeq ($(PROFILE), yes)
LDCLFLAGS += --profile_kernel data:all:all:all
endif

#Generates debug summary report
ifeq ($(DEBUG), yes)
CLFLAGS += --dk protocol:all:all:all
endif
