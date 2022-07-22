XLEN ?= 32

RISCV_TOOLCHAIN_PATH ?= /opt/riscv-gnu-toolchain
VORTEX_RT_PATH ?= $(realpath ../../../runtime)
VORTEX_KN_PATH ?= $(realpath ../../../kernel)

OPTS ?= -n64

VX_CC  = $(RISCV_TOOLCHAIN_PATH)/bin/riscv32-unknown-elf-gcc
VX_CXX = $(RISCV_TOOLCHAIN_PATH)/bin/riscv32-unknown-elf-g++
VX_DP  = $(RISCV_TOOLCHAIN_PATH)/bin/riscv32-unknown-elf-objdump
VX_CP  = $(RISCV_TOOLCHAIN_PATH)/bin/riscv32-unknown-elf-objcopy

VX_CFLAGS += -march=rv32imf -mabi=ilp32f -O3 -ffreestanding -nostartfiles -fdata-sections -ffunction-sections
VX_CFLAGS += -I$(VORTEX_KN_PATH)/include -I$(VORTEX_KN_PATH)/../hw

VX_LDFLAGS += -Wl,-Bstatic,-T,$(VORTEX_KN_PATH)/linker/vx_link$(XLEN).ld -Wl,--gc-sections $(VORTEX_KN_PATH)/libvortexrt.a

VX_SRCS = kernel.c

CXXFLAGS += -std=c++11 -Wall -Wextra -pedantic -Wfatal-errors

CXXFLAGS += -I$(VORTEX_RT_PATH)/include

LDFLAGS += -L$(VORTEX_RT_PATH)/stub -lvortex

# Debugigng
ifdef DEBUG
	CXXFLAGS += -g -O0
else    
	CXXFLAGS += -O2 -DNDEBUG
endif

PROJECT = fence

SRCS = main.cpp

all: $(PROJECT) kernel.bin kernel.dump
 
kernel.dump: kernel.elf
	$(VX_DP) -D kernel.elf > kernel.dump

kernel.bin: kernel.elf
	$(VX_CP) -O binary kernel.elf kernel.bin

kernel.elf: $(VX_SRCS)
	$(VX_CC) $(VX_CFLAGS) $(VX_SRCS) $(VX_LDFLAGS) -o kernel.elf

$(PROJECT): $(SRCS)
	$(CXX) $(CXXFLAGS) $^ $(LDFLAGS) -o $@

run-simx: $(PROJECT) kernel.bin   
	LD_LIBRARY_PATH=$(POCL_RT_PATH)/lib:$(VORTEX_RT_PATH)/simx:$(LD_LIBRARY_PATH) ./$(PROJECT) $(OPTS)
	
run-fpga: $(PROJECT) kernel.bin   
	LD_LIBRARY_PATH=$(POCL_RT_PATH)/lib:$(VORTEX_RT_PATH)/fpga:$(LD_LIBRARY_PATH) ./$(PROJECT) $(OPTS)

run-asesim: $(PROJECT) kernel.bin   
	LD_LIBRARY_PATH=$(POCL_RT_PATH)/lib:$(VORTEX_RT_PATH)/asesim:$(LD_LIBRARY_PATH) ./$(PROJECT) $(OPTS)
	
run-vlsim: $(PROJECT) kernel.bin   
	LD_LIBRARY_PATH=$(POCL_RT_PATH)/lib:$(VORTEX_RT_PATH)/vlsim:$(LD_LIBRARY_PATH) ./$(PROJECT) $(OPTS)

run-rtlsim: $(PROJECT) kernel.bin   
	LD_LIBRARY_PATH=$(POCL_RT_PATH)/lib:$(VORTEX_RT_PATH)/rtlsim:$(LD_LIBRARY_PATH) ./$(PROJECT) $(OPTS)

.depend: $(SRCS)
	$(CXX) $(CXXFLAGS) -MM $^ > .depend;

clean:
	rm -rf $(PROJECT) *.o .depend

clean-all: clean
	rm -rf *.elf *.bin *.dump

ifneq ($(MAKECMDGOALS),clean)
    -include .depend
endif