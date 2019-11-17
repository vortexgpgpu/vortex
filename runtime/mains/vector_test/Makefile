
COMP     = /home/fares/dev/riscv-gnu-toolchain-vector/drops/bin/riscv32-unknown-elf-gcc

CC_FLAGS = -ffreestanding -O0 -Wl,--gc-sections -nostartfiles -nostdlib -nostartfiles -nodefaultlibs -Wl,-Bstatic,-T,linker.ld -march=rv32imv -mabi=ilp32

DMP  = /home/fares/dev/riscv-gnu-toolchain-vector/drops/bin/riscv32-unknown-elf-objdump
CPY  = /home/fares/dev/riscv-gnu-toolchain-vector/drops/bin/riscv32-unknown-elf-objcopy

# VX_STR  = ../../startup/vx_start.s



NEWLIB  = ../../newlib/newlib.c
VX_STR  = ../../startup/vx_start.s
VX_INT  = ../../intrinsics/vx_intrinsics.s
VX_IO   = ../../io/vx_io.s ../../io/vx_io.c
VX_API  = ../../vx_api/vx_api.c
VX_TEST = ../../tests/tests.c
VX_FIO  = ../../fileio/fileio.s
VX_VEC  = vx_vec.s
LIBS    = /home/fares/dev/riscv-gnu-toolchain-vector/drops/riscv32-unknown-elf/lib/libc.a /home/fares/dev/riscv-gnu-toolchain-vector/drops/riscv32-unknown-elf/lib/libstdc++.a -static-libgcc -lgcc

VX_MAIN = vx_vector_main

all: HEX DUMP ELF

DUMP: ELF
	$(DMP) -D $(VX_MAIN).elf > $(VX_MAIN).dump

HEX: ELF
	$(CPY) -O ihex $(VX_MAIN).elf $(VX_MAIN).hex

ELF: 
	$(COMP) $(CC_FLAGS) $(VX_STR) $(VX_VEC) $(VX_FIO) $(NEWLIB) $(VX_INT) $(VX_IO) $(VX_API) $(VX_TEST) $(VX_MAIN).c $(LIBS) -Iinclude  -o $(VX_MAIN).elf
