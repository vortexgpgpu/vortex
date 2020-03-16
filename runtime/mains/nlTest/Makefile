
RISCV_TOOL_PATH ?= $(wildcard ../../../../riscv-gnu-toolchain/drops)

COMP = $(RISCV_TOOL_PATH)/bin/riscv32-unknown-elf-gcc
# CC_FLAGS = -march=rv32im -mabi=ilp32 -O0 -Wl,-Bstatic,-T,linker.ld -ffreestanding -nostdlib
CC_FLAGS = -march=rv32im -mabi=ilp32 -O0 -Wl,-Bstatic,-T,../vortex_link.ld -ffreestanding -nostartfiles -nostdlib

DMP  = $(RISCV_TOOL_PATH)/bin/riscv32-unknown-elf-objdump
CPY  = $(RISCV_TOOL_PATH)/bin/riscv32-unknown-elf-objcopy


NEWLIB  = ../../newlib/newlib.c
VX_STR  = ../../startup/vx_start.s
VX_INT  = ../../intrinsics/vx_intrinsics.s
VX_IO   = ../../io/vx_io.s ../../io/vx_io.c
VX_API  = ../../vx_api/vx_api.c
VX_TEST = ../../tests/tests.c
VX_FIO  = ../../fileio/fileio.s

VX_MAIN = ./vx_nl_main.c

all: HEX DUMP ELF

DUMP: ELF
	$(DMP) -D vx_nl_main.elf > vx_nl_main.dump

HEX: ELF
	$(CPY) -O ihex vx_nl_main.elf vx_nl_main.hex

ELF: 
	$(COMP) $(CC_FLAGS) $(VX_STR) $(VX_FIO) $(NEWLIB) $(VX_INT) $(VX_IO) $(VX_API) $(VX_TEST) $(VX_MAIN) $(RISCV_TOOL_PATH)/riscv32-unknown-elf/lib/libc.a $(RISCV_TOOL_PATH)/riscv32-unknown-elf/lib/libstdc++.a -static-libgcc -lgcc -o vx_nl_main.elf

clean:
	rm -rf *.o *.elf *.dump *.hex *.qemu *.log *.debug
