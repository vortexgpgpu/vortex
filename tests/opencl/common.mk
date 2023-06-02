XLEN ?= 32

TARGET ?= opaesim

XRT_SYN_DIR  ?= ../../../hw/syn/xilinx/xrt

ifeq ($(XLEN),64)
RISCV_TOOLCHAIN_PATH ?= /opt/riscv64-gnu-toolchain
VX_CFLAGS += -march=rv64imafd -mabi=lp64d
K_CFLAGS += -march=rv64imafd -mabi=ilp64d
K_LLCFLAGS += -O3 -march=riscv64 -target-abi=ilp64d -mcpu=generic-rv64
STARTUP_ADDR = 0x180000000
else
RISCV_TOOLCHAIN_PATH ?= /opt/riscv-gnu-toolchain
VX_CFLAGS += -march=rv32imaf -mabi=ilp32f
K_CFLAGS += -march=rv32imaf -mabi=ilp32f
K_LLCFLAGS += -O3 -march=riscv32 -target-abi=ilp32f -mcpu=generic-rv32
STARTUP_ADDR = 0x80000000
endif

RISCV_PREFIX ?= riscv$(XLEN)-unknown-elf
RISCV_SYSROOT ?= $(RISCV_TOOLCHAIN_PATH)/$(RISCV_PREFIX)

POCL_CC_PATH ?= /opt/pocl/compiler
POCL_RT_PATH ?= /opt/pocl/runtime

VORTEX_RT_PATH ?= $(realpath ../../../runtime)
VORTEX_KN_PATH ?= $(realpath ../../../kernel)

FPGA_BIN_DIR ?= $(VORTEX_RT_PATH)/opae

LLVM_VORTEX ?= /opt/llvm-vortex

K_LLCFLAGS += -O3 -mattr=+m,+f,+vortex -float-abi=hard
K_CFLAGS   += -v -O3 --sysroot=$(RISCV_SYSROOT) --gcc-toolchain=$(RISCV_TOOLCHAIN_PATH) -Xclang -target-feature -Xclang +vortex
K_CFLAGS   += -fno-rtti -fno-exceptions -nostartfiles -fdata-sections -ffunction-sections
K_CFLAGS   += -I$(VORTEX_KN_PATH)/include
K_LDFLAGS  += -Wl,-Bstatic,--gc-sections,-T$(VORTEX_KN_PATH)/linker/vx_link$(XLEN).ld,--defsym=STARTUP_ADDR=$(STARTUP_ADDR) $(VORTEX_KN_PATH)/libvortexrt.a -lm

CXXFLAGS += -std=c++11 -Wall -Wextra -Wfatal-errors
CXXFLAGS += -Wno-deprecated-declarations -Wno-unused-parameter -Wno-narrowing
CXXFLAGS += -I$(POCL_RT_PATH)/include

LDFLAGS += -L$(POCL_RT_PATH)/lib -L$(VORTEX_RT_PATH)/stub -lOpenCL -lvortex

# Debugigng
ifdef DEBUG
	CXXFLAGS += -g -O0
else    
	CXXFLAGS += -O2 -DNDEBUG
endif

ifeq ($(TARGET), fpga)
	OPAE_DRV_PATHS ?= libopae-c.so
else
ifeq ($(TARGET), asesim)
	OPAE_DRV_PATHS ?= libopae-c-ase.so
else
ifeq ($(TARGET), opaesim)
	OPAE_DRV_PATHS ?= libopae-c-sim.so
endif	
endif
endif

all: $(PROJECT) kernel.pocl
 
kernel.pocl: kernel.cl
	LLVM_PREFIX=$(LLVM_VORTEX) POCL_DEBUG=all LD_LIBRARY_PATH=$(LLVM_VORTEX)/lib:$(POCL_CC_PATH)/lib $(POCL_CC_PATH)/bin/poclcc -LLCFLAGS "$(K_LLCFLAGS)" -CFLAGS "$(K_CFLAGS)" -LDFLAGS "$(K_LDFLAGS)" -o kernel.pocl kernel.cl
 
$(PROJECT): $(SRCS)
	$(CXX) $(CXXFLAGS) $^ $(LDFLAGS) -o $@

run-simx: $(PROJECT) kernel.pocl   
	LD_LIBRARY_PATH=$(POCL_RT_PATH)/lib:$(VORTEX_RT_PATH)/simx:$(LD_LIBRARY_PATH) ./$(PROJECT) $(OPTS)

run-rtlsim: $(PROJECT) kernel.pocl   
	LD_LIBRARY_PATH=$(POCL_RT_PATH)/lib:$(VORTEX_RT_PATH)/rtlsim:$(LD_LIBRARY_PATH) ./$(PROJECT) $(OPTS)

run-opae: $(PROJECT) kernel.pocl
	SCOPE_JSON_PATH=$(FPGA_BIN_DIR)/scope.json OPAE_DRV_PATHS=$(OPAE_DRV_PATHS) LD_LIBRARY_PATH=$(POCL_RT_PATH)/lib:$(VORTEX_RT_PATH)/opae:$(LD_LIBRARY_PATH) ./$(PROJECT) $(OPTS)

run-xrt: $(PROJECT) kernel.pocl
ifeq ($(TARGET), hw)
	SCOPE_JSON_PATH=$(FPGA_BIN_DIR)/scope.json XRT_INI_PATH=$(XRT_SYN_DIR)/xrt.ini EMCONFIG_PATH=$(FPGA_BIN_DIR) XRT_DEVICE_INDEX=0 XRT_XCLBIN_PATH=$(FPGA_BIN_DIR)/vortex_afu.xclbin LD_LIBRARY_PATH=$(XILINX_XRT)/lib:$(POCL_RT_PATH)/lib:$(VORTEX_RT_PATH)/xrt:$(LD_LIBRARY_PATH) ./$(PROJECT) $(OPTS)
else
	XCL_EMULATION_MODE=$(TARGET) XRT_INI_PATH=$(XRT_SYN_DIR)/xrt.ini EMCONFIG_PATH=$(FPGA_BIN_DIR) XRT_DEVICE_INDEX=0 XRT_XCLBIN_PATH=$(FPGA_BIN_DIR)/vortex_afu.xclbin LD_LIBRARY_PATH=$(XILINX_XRT)/lib:$(POCL_RT_PATH)/lib:$(VORTEX_RT_PATH)/xrt:$(LD_LIBRARY_PATH) ./$(PROJECT) $(OPTS)	
endif

.depend: $(SRCS)
	$(CXX) $(CXXFLAGS) -MM $^ > .depend;

clean:
	rm -rf $(PROJECT) *.o .depend

clean-all: clean
	rm -rf *.elf *.bin *.dump *.pocl

ifneq ($(MAKECMDGOALS),clean)
    -include .depend
endif
