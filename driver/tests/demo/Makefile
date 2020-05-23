RISCV_TOOL_PATH ?= $(wildcard ~/dev/riscv-gnu-toolchain/drops)
VX_RT_PATH ?= $(wildcard ../../../runtime)

VX_CC  = $(RISCV_TOOL_PATH)/bin/riscv32-unknown-elf-gcc
VX_CXX = $(RISCV_TOOL_PATH)/bin/riscv32-unknown-elf-g++
VX_DMP = $(RISCV_TOOL_PATH)/bin/riscv32-unknown-elf-objdump
VX_CPY = $(RISCV_TOOL_PATH)/bin/riscv32-unknown-elf-objcopy

#VX_NEWLIB = $(VX_RT_PATH)/newlib/newlib.c
VX_STR = $(VX_RT_PATH)/startup/vx_start.S
VX_INT = $(VX_RT_PATH)/intrinsics/vx_intrinsics.S
#VX_IO  = $(VX_RT_PATH)/io/vx_io.S $(VX_RT_PATH)/io/vx_io.c
VX_API = $(VX_RT_PATH)/vx_api/vx_api.c
#VX_FIO = $(VX_RT_PATH)/fileio/fileio.S

VX_CFLAGS = -march=rv32im -mabi=ilp32 -O3 -Wl,-Bstatic,-T,$(VX_RT_PATH)/startup/vx_link.ld -ffreestanding -nostartfiles -Wl,--gc-sections

VX_SRCS = kernel.c

CXXFLAGS += -std=c++11 -O0 -g -Wall -Wextra -pedantic -Wfatal-errors

CXXFLAGS += -I../../include

PROJECT = demo

SRCS = demo.cpp

all: $(PROJECT) kernel.bin kernel.dump
 
kernel.dump:  kernel.elf
	$(VX_DMP) -D  kernel.elf > kernel.dump

kernel.bin:  kernel.elf
	$(VX_CPY) -O binary kernel.elf kernel.bin

kernel.elf: $(SRCS)
	$(VX_CC) $(VX_CFLAGS) $(VX_STR) $(VX_FIO) $(VX_NEWLIB) $(VX_INT) $(VX_IO) $(VX_API) $(VX_SRCS) -I$(VX_RT_PATH) -o kernel.elf

$(PROJECT): $(SRCS)
	$(CXX) $(CXXFLAGS) $^ $(LDFLAGS) -L../../stub -lvortex -o $@

run-fpga: $(PROJECT)
	LD_LIBRARY_PATH=../../opae:$(LD_LIBRARY_PATH) ./$(PROJECT) -f kernel.bin -n 16

run-ase: $(PROJECT)
	ASE_LOG=0 LD_LIBRARY_PATH=../../opae/ase:$(LD_LIBRARY_PATH) ./$(PROJECT) -f kernel.bin -n 16

run-rtlsim: $(PROJECT)
	LD_LIBRARY_PATH=../../rtlsim:$(LD_LIBRARY_PATH) ./$(PROJECT) -f kernel.bin -n 16
	
run-simx: $(PROJECT)
	LD_LIBRARY_PATH=../../simx:$(LD_LIBRARY_PATH) ./$(PROJECT) -f kernel.bin -n 16

.depend: $(SRCS)
	$(CXX) $(CXXFLAGS) -MM $^ > .depend;

clean:
	rm -rf $(PROJECT) *.o .depend

clean-all:
	rm -rf $(PROJECT) *.o *.elf *.bin *.dump .depend

ifneq ($(MAKECMDGOALS),clean)
    -include .depend
endif