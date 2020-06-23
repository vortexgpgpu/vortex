
RISCV_TOOL_PATH ?= $(wildcard ../../../../riscv-gnu-toolchain/drops)

COMP     = $(RISCV_TOOL_PATH)/bin/riscv32-unknown-elf-gcc
# CC_FLAGS = -march=rv32im -mabi=ilp32 -O0 -Wl,-Bstatic,-T,linker.ld -ffreestanding -nostdlib
CC_FLAGS = -march=rv32im -mabi=ilp32 -O0 -Wl,-Bstatic,-T,../../startup/vx_link.ld

DMP  = $(RISCV_TOOL_PATH)/bin/riscv32-unknown-elf-objdump
CPY  = $(RISCV_TOOL_PATH)/bin/riscv32-unknown-elf-objcopy


NEWLIB  = ../../newlib/newlib.c
VX_STR  = 
VX_INT  = ../../intrinsics/vx_intrinsics.S
VX_IO   = 
VX_API  = 
VX_FIO  = 
LIBS    = $(RISCV_TOOL_PATH)/riscv32-unknown-elf/lib/libc.a $(RISCV_TOOL_PATH)/riscv32-unknown-elf/lib/libstdc++.a -static-libgcc -lgcc

VX_MAIN = hello

all: HEX DUMP ELF

DUMP: ELF
	$(DMP) -D $(VX_MAIN).elf > $(VX_MAIN).dump

HEX: ELF
	$(CPY) -O ihex $(VX_MAIN).elf $(VX_MAIN).hex

ELF: 
	$(COMP) $(CC_FLAGS) $(VX_STR) $(VX_FIO) $(NEWLIB) $(VX_INT) $(VX_IO) $(VX_API) $(VX_MAIN).cpp $(LIBS) -Iinclude -o $(VX_MAIN).elf
