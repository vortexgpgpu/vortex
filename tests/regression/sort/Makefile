
RISCV_TOOLCHAIN_PATH ?= /opt/riscv-gnu-toolchain
VORTEX_DRV_PATH ?= $(realpath ../../../driver)
VORTEX_RT_PATH ?= $(realpath ../../../runtime)
VORTEX_HW_PATH ?= $(realpath ../../../hw)
LLVM_PREFIX ?= /opt/llvm-riscv
SYSROOT=${RISCV_TOOLCHAIN_PATH}/riscv32-unknown-elf

OPTS ?= -n16

VX_CC  = ${LLVM_PREFIX}/bin/clang
VX_CXX = ${LLVM_PREFIX}/bin/clang++
VX_DP  = ${LLVM_PREFIX}/bin/llvm-objdump
VX_CP  = ${LLVM_PREFIX}/bin/llvm-objcopy

VX_CFLAGS += -O3 -march=rv32imf -mabi=ilp32f -fno-rtti -fno-exceptions -ffreestanding -nostartfiles -fdata-sections -ffunction-sections
VX_CFLAGS += -Xclang -target-feature -Xclang +vortex
VX_CFLAGS += --sysroot=${SYSROOT} --gcc-toolchain=${RISCV_TOOLCHAIN_PATH}
VX_CFLAGS += -I${VORTEX_HW_PATH} -I${VORTEX_RT_PATH}/include 

VX_LDFLAGS += -Wl,-Bstatic,-T${VORTEX_RT_PATH}/linker/vx_link.ld,--gc-sections ${VORTEX_RT_PATH}/libvortexrt.a

VX_DPFLAGS = -arch=riscv32 -mcpu=generic-rv32 -mattr=+m,+f -mattr=+vortex

VX_SRCS = kernel.c

#CXXFLAGS += -std=c++11 -O2 -Wall -Wextra -pedantic -Wfatal-errors
CXXFLAGS += -std=c++11 -O0 -g -Wall -Wextra -pedantic -Wfatal-errors

CXXFLAGS += -I$(VORTEX_DRV_PATH)/include

LDFLAGS += -L$(VORTEX_DRV_PATH)/stub -lvortex

PROJECT = sort

SRCS = main.cpp

all: $(PROJECT) kernel.bin kernel.dump
 
kernel.dump: kernel.elf
	$(VX_DP) $(VX_DPFLAGS) -D kernel.elf > kernel.dump

kernel.bin: kernel.elf
	$(VX_CP) -O binary kernel.elf kernel.bin

kernel.elf: $(VX_SRCS)
	$(VX_CC) $(VX_CFLAGS) $(VX_SRCS) $(VX_LDFLAGS) -o kernel.elf

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