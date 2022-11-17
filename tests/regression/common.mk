XLEN ?= 32

TARGET   ?= hw_emu
PLATFORM ?= xilinx_u280_xdma_201920_3

XRT_SYN_DIR  ?= ../../../hw/syn/xilinx/xrt
XRT_BUILD_DIR = ${XRT_SYN_DIR}/build_${PLATFORM}_${TARGET}/bin

RISCV_TOOLCHAIN_PATH ?= /opt/riscv-gnu-toolchain

VORTEX_RT_PATH ?= $(realpath ../../../runtime)
VORTEX_KN_PATH ?= $(realpath ../../../kernel)

LLVM_PREFIX ?= /opt/llvm-riscv

LLVM_CFLAGS += --sysroot=${RISCV_TOOLCHAIN_PATH}/riscv32-unknown-elf
LLVM_CFLAGS += --gcc-toolchain=${RISCV_TOOLCHAIN_PATH}
LLVM_CFLAGS += -Xclang -target-feature -Xclang +vortex
#LLVM_CFLAGS += -I$(RISCV_TOOLCHAIN_PATH)/riscv32-unknown-elf/include/c++/9.2.0/riscv32-unknown-elf 
#LLVM_CFLAGS += -I$(RISCV_TOOLCHAIN_PATH)/riscv32-unknown-elf/include/c++/9.2.0
#LLVM_CFLAGS += -Wl,-L$(RISCV_TOOLCHAIN_PATH)/lib/gcc/riscv32-unknown-elf/9.2.0
#LLVM_CFLAGS += --rtlib=libgcc

VX_CC  = $(LLVM_PREFIX)/bin/clang $(LLVM_CFLAGS)
VX_CXX = $(LLVM_PREFIX)/bin/clang++ $(LLVM_CFLAGS)
VX_DP  = $(LLVM_PREFIX)/bin/llvm-objdump -mattr=+m,+f,+vortex
VX_CP  = $(LLVM_PREFIX)/bin/llvm-objcopy

#VX_CC  = $(RISCV_TOOLCHAIN_PATH)/bin/riscv32-unknown-elf-gcc
#VX_CXX = $(RISCV_TOOLCHAIN_PATH)/bin/riscv32-unknown-elf-g++
#VX_DP  = $(RISCV_TOOLCHAIN_PATH)/bin/riscv32-unknown-elf-objdump
#VX_CP  = $(RISCV_TOOLCHAIN_PATH)/bin/riscv32-unknown-elf-objcopy

VX_CFLAGS += -v -O3 -std=c++17 -march=rv32imf -mabi=ilp32f
VX_CFLAGS += -fno-rtti -fno-exceptions -nostartfiles -fdata-sections -ffunction-sections
VX_CFLAGS += -I$(VORTEX_KN_PATH)/include -I$(VORTEX_KN_PATH)/../hw

VX_LDFLAGS += -Wl,-Bstatic,-T,$(VORTEX_KN_PATH)/linker/vx_link$(XLEN).ld -Wl,--gc-sections $(VORTEX_KN_PATH)/libvortexrt.a

CXXFLAGS += -std=c++17 -Wall -Wextra -pedantic -Wfatal-errors

CXXFLAGS += -I$(VORTEX_RT_PATH)/include -I$(VORTEX_KN_PATH)/../hw

LDFLAGS += -L$(VORTEX_RT_PATH)/stub -lvortex

# Debugigng
ifdef DEBUG
	CXXFLAGS += -g -O0
else    
	CXXFLAGS += -O2 -DNDEBUG
endif

ifeq ($(TARGET), fpga)
	OPAE_DRV_PATHS ?= $(OPAE_SDK_ROOT)/lib64/libopae-c.so
else
ifeq ($(TARGET), asesim)
	OPAE_DRV_PATHS ?= $(OPAE_SDK_ROOT)/lib64/libopae-c-ase.so
else
	OPAE_DRV_PATHS ?= ../../../sim/opaesim/libopae-c-sim.so
endif
endif

all: $(PROJECT) kernel.bin kernel.dump
 
kernel.dump: kernel.elf
	$(VX_DP) -D kernel.elf > kernel.dump

kernel.bin: kernel.elf
	$(VX_CP) -O binary kernel.elf kernel.bin

kernel.elf: $(VX_SRCS)
	$(VX_CXX) $(VX_CFLAGS) $(VX_SRCS) $(VX_LDFLAGS) -o kernel.elf

$(PROJECT): $(SRCS)
	$(CXX) $(CXXFLAGS) $^ $(LDFLAGS) -o $@

run-simx: $(PROJECT) kernel.bin   
	LD_LIBRARY_PATH=$(VORTEX_RT_PATH)/simx:$(LD_LIBRARY_PATH) ./$(PROJECT) $(OPTS)
	
run-opae: $(PROJECT) kernel.bin   
	OPAE_DRV_PATHS=$(OPAE_DRV_PATHS) LD_LIBRARY_PATH=$(VORTEX_RT_PATH)/opae:$(LD_LIBRARY_PATH) ./$(PROJECT) $(OPTS)

run-rtlsim: $(PROJECT) kernel.bin   
	LD_LIBRARY_PATH=$(VORTEX_RT_PATH)/rtlsim:$(LD_LIBRARY_PATH) ./$(PROJECT) $(OPTS)

run-xrt: $(PROJECT) kernel.bin
ifeq ($(TARGET), hw)
	XRT_INI_PATH=${XRT_SYN_DIR}/xrt.ini EMCONFIG_PATH=${XRT_BUILD_DIR} XRT_DEVICE_INDEX=0 XRT_XCLBIN_PATH=${XRT_BUILD_DIR}/vortex_afu.xclbin LD_LIBRARY_PATH=$(XILINX_XRT)/lib:$(VORTEX_RT_PATH)/xrt:$(LD_LIBRARY_PATH) ./$(PROJECT) $(OPTS)
else
	XCL_EMULATION_MODE=${TARGET} XRT_INI_PATH=${XRT_SYN_DIR}/xrt.ini EMCONFIG_PATH=${XRT_BUILD_DIR} XRT_DEVICE_INDEX=0 XRT_XCLBIN_PATH=${XRT_BUILD_DIR}/vortex_afu.xclbin LD_LIBRARY_PATH=$(XILINX_XRT)/lib:$(VORTEX_RT_PATH)/xrt:$(LD_LIBRARY_PATH) ./$(PROJECT) $(OPTS)	
endif

.depend: $(SRCS)
	$(CXX) $(CXXFLAGS) -MM $^ > .depend;

clean:
	rm -rf $(PROJECT) *.o .depend

clean-all: clean
	rm -rf *.elf *.bin *.dump

ifneq ($(MAKECMDGOALS),clean)
    -include .depend
endif
