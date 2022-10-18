ifneq ($(findstring Makefile, $(MAKEFILE_LIST)), Makefile)
help:
	$(ECHO) "Makefile Usage:"
	$(ECHO) "  make all TARGET=<sw_emu/hw_emu/hw> PLATFORM=<FPGA platform>"
	$(ECHO) "      Command to generate the design for specified Target and Device."
	$(ECHO) ""
	$(ECHO) "  make clean"
	$(ECHO) "      Command to remove the generated non-hardware files."
	$(ECHO) ""
endif

TARGET ?= hw
PLATFORM ?=

PREFIX ?= build

RTL_DIR = ../../../../rtl
AFU_DIR = ../../../../afu/xrt
IP_DIR  = ../../../../ip/xilinx

BUILD_DIR = $(PREFIX)_$(PLATFORM)_$(TARGET)
BIN_DIR   = $(BUILD_DIR)/bin

# Control RTL debug tracing states
DBG_TRACE_FLAGS += -DDBG_TRACE_CORE_PIPELINE  
DBG_TRACE_FLAGS += -DDBG_TRACE_CORE_ICACHE
DBG_TRACE_FLAGS += -DDBG_TRACE_CORE_DCACHE
DBG_TRACE_FLAGS += -DDBG_TRACE_CORE_MEM
DBG_TRACE_FLAGS += -DDBG_TRACE_CACHE_BANK 
DBG_TRACE_FLAGS += -DDBG_TRACE_CACHE_MSHR
DBG_TRACE_FLAGS += -DDBG_TRACE_CACHE_TAG
DBG_TRACE_FLAGS += -DDBG_TRACE_CACHE_DATA
DBG_TRACE_FLAGS += -DDBG_TRACE_AFU
DBG_TRACE_FLAGS += -DDBG_TRACE_TEX
DBG_TRACE_FLAGS += -DDBG_TRACE_RASTER
DBG_TRACE_FLAGS += -DDBG_TRACE_ROP

DBG_FLAGS += $(DBG_TRACE_FLAGS)

FPU_INCLUDE = -I$(RTL_DIR)/fpu_unit
TEX_INCLUDE = -I$(RTL_DIR)/tex_unit
RASTER_INCLUDE = -I$(RTL_DIR)/raster_unit
ROP_INCLUDE = -I$(RTL_DIR)/rop_unit
RTL_INCLUDE = -I$(RTL_DIR) -I$(RTL_DIR)/libs -I$(RTL_DIR)/interfaces -I$(RTL_DIR)/cache -I$(AFU_DIR) -I$(IP_DIR)
RTL_INCLUDE += $(FPU_INCLUDE) $(TEX_INCLUDE) $(RASTER_INCLUDE) $(ROP_INCLUDE)

#CONFIGS += -DEXT_GFX_ENABLE
#CONFIGS += -DNUM_WARPS=2 -DNUM_THREADS=2 -DEXT_F_DISABLE -DL1_DISABLE
#CONFIGS += -DNUM_WARPS=2 -DNUM_THREADS=2 -DEXT_F_DISABLE
#CONFIGS += -DNUM_WARPS=2 -DNUM_THREADS=2
#CONFIGS += -DNUM_CORES=4

CONFIGS += -DSYNTHESIS -DVIVADO

VIVADO = $(XILINX_VIVADO)/bin/vivado

VPP = $(XILINX_VITIS)/bin/v++

CP = cp -rf
RMDIR = rm -rf

ECHO = @echo

XO_CONTAINER = $(BIN_DIR)/vortex_afu.xo

XCLBIN_CONTAINER = $(BIN_DIR)/vortex_afu.xclbin

NCPUS := $(shell grep -c ^processor /proc/cpuinfo)
JOBS := $(shell expr $(NCPUS) - 1)

# Kernel compiler global settings
VPP_FLAGS += --link --target $(TARGET) --platform $(PLATFORM) --save-temps --no_ip_cache
VPP_FLAGS += --vivado.synth.jobs $(JOBS) --vivado.impl.jobs $(JOBS)
VPP_FLAGS += --connectivity.sp vortex_afu_1.m_axi_mem:HBM[0:15]
VPP_FLAGS += --report_level 2
VPP_FLAGS += --config ../vitis.ini

# Enable perf counters
ifdef PERF
	CONFIGS += -DPERF_ENABLE
endif

# Generates profile summary report
ifdef PROFILE
	VPP_FLAGS += --profile_kernel data:all:all:all
	VPP_FLAGS += --profile_kernel stall:all:all:all
endif

# Debugigng
ifdef DEBUG
	VPP_FLAGS += -g --optimize 0 --debug.protocol all
	ifeq ($(TARGET), hw)
		VPP_FLAGS += --debug.chipscope vortex_afu_1
		CONFIGS += -DCHIPSCOPE
	else
		CONFIGS += -DSIMULATION $(DBG_FLAGS)
		VPP_FLAGS += --vivado.prop fileset.sim_1.xsim.elaborate.debug_level=all
	endif
else    
	CONFIGS += -DNDEBUG
	VPP_FLAGS += --optimize 3
endif

# RTL Kernel only supports Hardware and Hardware Emulation.
ifneq ($(TARGET),$(findstring $(TARGET), hw hw_emu))
	$(warning WARNING:Application supports only hw hw_emu TARGET. Please use the target for running the application)
endif

.PHONY: all clean gen-sources emconfig check-devices

all: check-devices $(XCLBIN_CONTAINER) emconfig

gen-sources: $(BUILD_DIR)/sources.txt
$(BUILD_DIR)/sources.txt:
	mkdir -p $(BUILD_DIR); cd $(BUILD_DIR); ../scripts/gen_sources.sh $(RTL_INCLUDE) $(CONFIGS) > sources.txt

$(XO_CONTAINER): $(BUILD_DIR)/sources.txt ./kernel.xml
	mkdir -p $(BUILD_DIR); cd $(BUILD_DIR);	$(VIVADO) -mode batch -source ../scripts/gen_xo.tcl -tclargs ../$(XO_CONTAINER) vortex_afu sources.txt ../kernel.xml ../$(BUILD_DIR)

$(XCLBIN_CONTAINER): $(XO_CONTAINER)
	mkdir -p $(BIN_DIR); cd $(BUILD_DIR); $(VPP) $(VPP_FLAGS) -o ../$(XCLBIN_CONTAINER) ../$(XO_CONTAINER)

emconfig: $(BIN_DIR)/emconfig.json
$(BIN_DIR)/emconfig.json:
	mkdir -p $(BIN_DIR); cd $(BUILD_DIR); emconfigutil --platform $(PLATFORM) --od ../$(BIN_DIR)

hw_server:
	debug_hw --xvc_pcie /dev/xfpga/xvc_pub.u2305.0 --hw_server &

chipscope:	
	debug_hw --vivado --host localhost --ltx_file $(BUILD_DIR)/_x/link/vivado/vpl/prj/prj.runs/impl_1/debug_nets.ltx &

clean:
	$(RMDIR) $(BUILD_DIR)

# Check the devices avaiable
check-devices:
ifndef PLATFORM
	$(error PLATFORM not set. Please set the PLATFORM properly and rerun. Run "make help" for more details.)
endif
ifndef XILINX_VITIS
	$(error XILINX_VITIS variable is not set, please set correctly and rerun)
endif
