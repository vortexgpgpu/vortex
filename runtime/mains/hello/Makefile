
COMP     = ~/dev/riscv-gnu-toolchain/drops/bin/riscv32-unknown-elf-gcc
# CC_FLAGS = -march=rv32im -mabi=ilp32 -O0 -Wl,-Bstatic,-T,linker.ld -ffreestanding -nostdlib
CC_FLAGS = -march=rv32im -mabi=ilp32 -O0 -Wl,-Bstatic,-T,../vortex_link.ld

DMP  = ~/dev/riscv-gnu-toolchain/drops/bin/riscv32-unknown-elf-objdump
CPY  = ~/dev/riscv-gnu-toolchain/drops/bin/riscv32-unknown-elf-objcopy


NEWLIB  = ../../newlib/newlib.c ../../newlib/newlib_notimp.c ../../newlib/newlib.s
VX_STR  = 
VX_INT  = ../../intrinsics/vx_intrinsics.s
VX_IO   = 
VX_API  = 
VX_TEST = 
VX_FIO  = 
LIBS    = ~/dev/riscv-gnu-toolchain/drops/riscv32-unknown-elf/lib/libc.a ~/dev/riscv-gnu-toolchain/drops/riscv32-unknown-elf/lib/libstdc++.a -static-libgcc -lgcc

VX_MAIN = hello

all: HEX DUMP ELF

DUMP: ELF
	$(DMP) -D $(VX_MAIN).elf > $(VX_MAIN).dump

HEX: ELF
	$(CPY) -O ihex $(VX_MAIN).elf $(VX_MAIN).hex

ELF: 
	$(COMP) $(CC_FLAGS) $(VX_STR) $(VX_FIO) $(NEWLIB) $(VX_INT) $(VX_IO) $(VX_API) $(VX_TEST) $(VX_MAIN).cpp $(LIBS) -Iinclude -o $(VX_MAIN).elf
