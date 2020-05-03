
COMP     = ~/dev/riscv-gnu-toolchain/drops/bin/riscv32-unknown-elf-g++
#COMP     = /opt/riscv-new/drops/bin/riscv32-unknown-elf-g++

CC_FLAGS = -march=rv32im -mabi=ilp32 -O3 -Wl,-Bstatic,-T,../../startup/vx_link.ld -ffreestanding -nostdlib

DMP  = ~/dev/riscv-gnu-toolchain/drops/bin/riscv32-unknown-elf-objdump
CPY  = ~/dev/riscv-gnu-toolchain/drops/bin/riscv32-unknown-elf-objcopy


NEWLIB  = ../../newlib/newlib.c
VX_STR  = ../../startup/vx_start.S
VX_INT  = ../../intrinsics/vx_intrinsics.S
VX_IO   = ../../io/vx_io.S ../../io/vx_io.c
VX_API  = ../../vx_api/vx_api.c
VX_FIO  = ../../fileio/fileio.S

VX_MAIN = vx_simple_main

LIBS    = ~/dev/riscv-gnu-toolchain/drops/riscv32-unknown-elf/lib/libc.a ~/dev/riscv-gnu-toolchain/drops/riscv32-unknown-elf/lib/libstdc++.a -static-libgcc -lgcc
#LIBS    = /opt/riscv-new/drops/riscv32-unknown-elf/lib/libc.a /opt/riscv-new/drops/riscv32-unknown-elf/lib/libstdc++.a -static-libgcc -lgcc

VX_SRCS = vx_simple_main.c tests.c

all: HEX DUMP ELF

DUMP: ELF
	$(DMP) -D $(VX_MAIN).elf > $(VX_MAIN).dump

HEX: ELF
	$(CPY) -O ihex $(VX_MAIN).elf $(VX_MAIN).hex

ELF: 
	$(COMP) $(CC_FLAGS) $(VX_STR) $(VX_FIO) $(NEWLIB) $(VX_INT) $(VX_IO) $(VX_API) $(VX_SRCS) $(LIBS) -Iinclude  -o $(VX_MAIN).elf