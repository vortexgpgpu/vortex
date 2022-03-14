XLEN ?= 32

RISCV_TOOLCHAIN_PATH ?= /opt/riscv-gnu-toolchain
VORTEX_DRV_PATH ?= $(realpath ../../../driver)
VORTEX_RT_PATH ?= $(wildcard ../../../runtime)
LLVM_PREFIX ?= /opt/llvm-riscv

OPTS ?= 

VX_CC  = $(RISCV_TOOLCHAIN_PATH)/bin/riscv32-unknown-elf-gcc
VX_CXX = $(RISCV_TOOLCHAIN_PATH)/bin/riscv32-unknown-elf-g++
VX_DP  = $(RISCV_TOOLCHAIN_PATH)/bin/riscv32-unknown-elf-objdump
VX_CP  = $(RISCV_TOOLCHAIN_PATH)/bin/riscv32-unknown-elf-objcopy

#VX_CC  = ${LLVM_PREFIX}/bin/clang
#VX_CXX = ${LLVM_PREFIX}/bin/clang++
#VX_DP  = ${LLVM_PREFIX}/bin/llvm-objdump
#VX_CP  = ${LLVM_PREFIX}/bin/llvm-objcopy

VX_CFLAGS += -std=c++17 -march=rv32imf -mabi=ilp32f -O3 -ffreestanding -nostartfiles -fdata-sections -ffunction-sections
VX_CFLAGS += -DENABLE_SW -I$(VORTEX_RT_PATH)/include -I$(VORTEX_RT_PATH)/../hw -I$(VORTEX_RT_PATH)/../sim/common -I$(VORTEX_RT_PATH)/../third_party

VX_LDFLAGS += -Wl,-Bstatic,-T,$(VORTEX_RT_PATH)/linker/vx_link$(XLEN).ld -Wl,--gc-sections $(VORTEX_RT_PATH)/libvortexrt.a

VX_SRCS = kernel.cpp

CXXFLAGS += -std=c++17 -Wall -Wextra -Wfatal-errors

CXXFLAGS += -I$(VORTEX_DRV_PATH)/include -I$(VORTEX_RT_PATH)/../hw -I$(VORTEX_RT_PATH)/../sim/common -I$(VORTEX_RT_PATH)/../third_party

LDFLAGS += -L$(VORTEX_DRV_PATH)/stub -lvortex $(VORTEX_RT_PATH)/../third_party/cocogfx/libcocogfx.a -lpng -lz -lboost_serialization

# Debugigng
ifdef DEBUG
	CXXFLAGS += -g -O0
else    
	CXXFLAGS += -O2 -DNDEBUG
endif

PROJECT = draw3d

SRCS = main.cpp utils.cpp

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
	LD_LIBRARY_PATH=$(POCL_RT_PATH)/lib:$(VORTEX_DRV_PATH)/simx:$(LD_LIBRARY_PATH) ./$(PROJECT) $(OPTS)
	
run-fpga: $(PROJECT) kernel.bin   
	LD_LIBRARY_PATH=$(POCL_RT_PATH)/lib:$(VORTEX_DRV_PATH)/fpga:$(LD_LIBRARY_PATH) ./$(PROJECT) $(OPTS)

run-asesim: $(PROJECT) kernel.bin   
	LD_LIBRARY_PATH=$(POCL_RT_PATH)/lib:$(VORTEX_DRV_PATH)/asesim:$(LD_LIBRARY_PATH) ./$(PROJECT) $(OPTS)
	
run-vlsim: $(PROJECT) kernel.bin   
	LD_LIBRARY_PATH=$(POCL_RT_PATH)/lib:$(VORTEX_DRV_PATH)/vlsim:$(LD_LIBRARY_PATH) ./$(PROJECT) $(OPTS)

run-rtlsim: $(PROJECT) kernel.bin   
	LD_LIBRARY_PATH=$(POCL_RT_PATH)/lib:$(VORTEX_DRV_PATH)/rtlsim:$(LD_LIBRARY_PATH) ./$(PROJECT) $(OPTS)

.depend: $(SRCS)
	$(CXX) $(CXXFLAGS) -MM $^ > .depend;

clean:
	rm -rf $(PROJECT) *.o .depend

clean-all: clean
	rm -rf *.elf *.bin *.dump

ifneq ($(MAKECMDGOALS),clean)
    -include .depend
endif