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

TARGET := hw
PLATFORM :=
PREFIX := build

RTL_DIR = ../../../rtl
AFU_DIR = ../../../afu/xrt
IP_DIR  = ../../../ip/xilinx

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

CONFIGS += -DEXT_GFX_ENABLE
CONFIGS += -DSYNTHESIS -DVIVADO -DNDEBUG -DMEM_BLOCK_SIZE=16

# Debugigng
ifdef DEBUG
	CONFIGS += $(DBG_FLAGS)
else    
	CONFIGS += -DNDEBUG
endif

# Enable perf counters
ifdef PERF
	CONFIGS += -DPERF_ENABLE
endif

VIVADO = $(XILINX_VIVADO)/bin/vivado

VPP = $(XILINX_VITIS)/bin/v++

CP = cp -rf
RMDIR = rm -rf

ECHO = @echo

BUILD_DIR = $(PREFIX)_$(PLATFORM)_$(TARGET)
BIN_DIR = $(BUILD_DIR)/bin

XO_CONTAINER = $(BIN_DIR)/vortex_afu.xo

XCLBIN_CONTAINER = $(BIN_DIR)/vortex_afu.xclbin

# Kernel compiler global settings
VPP_CFLAGS += -t $(TARGET) --platform $(PLATFORM) --save-temps
VPP_CFLAGS += --temp_dir $(BUILD_DIR) --log_dir $(BUILD_DIR) --report_dir $(BUILD_DIR)
VPP_LDFLAGS +=

.PHONY: all clean gen-sources emconfig

all: check-devices $(XCLBIN_CONTAINER) emconfig

gen-sources: $(BUILD_DIR)/sources.txt
$(BUILD_DIR)/sources.txt:
	mkdir -p $(BUILD_DIR)
	./scripts/gen_sources.sh $(RTL_INCLUDE) $(CONFIGS) > $(BUILD_DIR)/sources.txt

$(XO_CONTAINER): $(BUILD_DIR)/sources.txt
	mkdir -p $(BUILD_DIR)
	$(VIVADO) -log $(BUILD_DIR)/vivado.log -journal $(BUILD_DIR)/vivado.jou -tempDir $(BUILD_DIR) -mode batch -source scripts/gen_xo.tcl -tclargs $(XO_CONTAINER) vortex_afu $(BUILD_DIR)/sources.txt $(BUILD_DIR)

$(XCLBIN_CONTAINER): $(XO_CONTAINER)
	mkdir -p $(BIN_DIR)
	$(VPP) $(VPP_CFLAGS) $(VPP_LDFLAGS) -lo $(XCLBIN_CONTAINER) $(XO_CONTAINER)

emconfig: $(BIN_DIR)/emconfig.json
$(BIN_DIR)/emconfig.json:
	mkdir -p $(BIN_DIR)
	emconfigutil --platform $(PLATFORM) --od $(BIN_DIR)

clean:
	-$(RMDIR) $(BUILD_DIR) .Xil *.log

# RTL Kernel only supports Hardware and Hardware Emulation.
ifneq ($(TARGET),$(findstring $(TARGET), hw hw_emu))
	$(warning WARNING:Application supports only hw hw_emu TARGET. Please use the target for running the application)
endif

# Check the devices avaiable
check-devices:
ifndef PLATFORM
	$(error PLATFORM not set. Please set the PLATFORM properly and rerun. Run "make help" for more details.)
endif
ifndef XILINX_VITIS
	$(error XILINX_VITIS variable is not set, please set correctly and rerun)
endif

# Enable profiling if needed
REPORT := no
PROFILE := no
DEBUG := no

#'estimate' for estimate report generation
#'system' for system report generation
ifneq ($(REPORT), no)
VPP_CFLAGS += --report estimate
endif

# Generates profile summary report
ifeq ($(PROFILE), yes)
VPP_CFLAGS += --profile_kernel data:all:all:all
endif

# Generates debug summary report
ifeq ($(DEBUG), yes)
VPP_CFLAGS += --dk protocol:all:all:all
VPP_CFLAGS += --dk list_ports
endif
