
COMP     = /opt/riscv-new/drops/bin/riscv32-unknown-elf-gcc
CC_FLAGS = -march=rv32im -mabi=ilp32 -O0 -Wl,-Bstatic,-T,../vortex_link.ld -ffreestanding -nostdlib

DMP  = /opt/riscv-new/drops/bin/riscv32-unknown-elf-objdump
CPY  = /opt/riscv-new/drops/bin/riscv32-unknown-elf-objcopy


NEWLIB  = ../../newlib/newlib.c
VX_STR  = ../../startup/vx_start.s
VX_INT  = ../../intrinsics/vx_intrinsics.s
VX_IO   = ../../io/vx_io.s ../../io/vx_io.c
VX_API  = ../../vx_api/vx_api.c
VX_TEST = ../../tests/tests.c
VX_FIO  = ../../fileio/fileio.s
LIBS    = /opt/riscv-new/drops/riscv32-unknown-elf/lib/libc.a /opt/riscv-new/drops/riscv32-unknown-elf/lib/libstdc++.a -static-libgcc -lgcc

VX_MAIN = vx_simple_main

all: HEX DUMP ELF

DUMP: ELF
	$(DMP) -D $(VX_MAIN).elf > $(VX_MAIN).dump

HEX: ELF
	$(CPY) -O ihex $(VX_MAIN).elf $(VX_MAIN).hex

ELF: 
	$(COMP) $(CC_FLAGS) $(VX_STR) $(VX_FIO) $(NEWLIB) $(VX_INT) $(VX_IO) $(VX_API) $(VX_TEST) $(VX_MAIN).c $(LIBS) -Iinclude  -o $(VX_MAIN).elf