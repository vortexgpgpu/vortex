RISCV_TOOL_PATH ?= $(wildcard ~/dev/riscv-gnu-toolchain/drops)
VX_RT_PATH ?= $(wildcard ../../../runtime)

VX_CC  = $(RISCV_TOOL_PATH)/bin/riscv32-unknown-elf-gcc
VX_CXX = $(RISCV_TOOL_PATH)/bin/riscv32-unknown-elf-g++
VX_DMP = $(RISCV_TOOL_PATH)/bin/riscv32-unknown-elf-objdump
VX_CPY = $(RISCV_TOOL_PATH)/bin/riscv32-unknown-elf-objcopy

VX_NEWLIB = $(VX_RT_PATH)/newlib/newlib.c
VX_STR = $(VX_RT_PATH)/startup/vx_start.S
VX_INT = $(VX_RT_PATH)/intrinsics/vx_intrinsics.s
VX_IO  = $(VX_RT_PATH)/io/vx_io.s $(VX_RT_PATH)/io/vx_io.c
VX_API = $(VX_RT_PATH)/vx_api/vx_api.c
VX_FIO = $(VX_RT_PATH)/fileio/fileio.s

VX_CFLAGS = -march=rv32im -mabi=ilp32 -O3 -Wl,-Bstatic,-T,$(VX_RT_PATH)/mains/vortex_link.ld -ffreestanding -nostartfiles -Wl,--gc-sections

VX_SRCS = kernel.c

CXXFLAGS += -std=c++11 -O0 -g -Wall -Wextra -pedantic -Wfatal-errors

CXXFLAGS += -I../../sw/include

PROJECT = demo

SRCS = demo.cpp

all: $(PROJECT)
 
kernel.dump:  kernel.elf
	$(VX_DMP) -D  kernel.elf > kernel.dump

kernel.hex:  kernel.elf
	$(VX_CPY) -O ihex kernel.elf kernel.hex

kernel.bin:  kernel.elf
	$(VX_CPY) -O binary kernel.elf kernel.bin

kernel.elf: $(SRCS)
	$(VX_CC) $(VX_CFLAGS) $(VX_STR) $(VX_FIO) $(VX_NEWLIB) $(VX_INT) $(VX_IO) $(VX_API) $(VX_SRCS) -I$(VX_RT_PATH) -o kernel.elf

$(PROJECT): $(SRCS)
	$(CXX) $(CXXFLAGS) $^ $(LDFLAGS) -L../../sw/simx -lvortex -o $@

run-fpga: $(PROJECT)
	LD_LIBRARY_PATH=../../sw/opae:$(LD_LIBRARY_PATH) ./$(PROJECT) -f kernel.bin

run-ase: $(PROJECT)
	LD_LIBRARY_PATH=../../sw/opae/ase:$(LD_LIBRARY_PATH) ./$(PROJECT) -f kernel.bin

run-rtlsim: $(PROJECT)
	LD_LIBRARY_PATH=../../sw/rtlsim:$(LD_LIBRARY_PATH) ./$(PROJECT) -f kernel.bin
	
run-simx: $(PROJECT)
	LD_LIBRARY_PATH=../../sw/simx:$(LD_LIBRARY_PATH) ./$(PROJECT) -f kernel.bin

.depend: $(SRCS)
	$(CXX) $(CXXFLAGS) -MM $^ > .depend;

clean:
	rm -rf $(PROJECT) *.o *.dump .depend 

ifneq ($(MAKECMDGOALS),clean)
    -include .depend
endif