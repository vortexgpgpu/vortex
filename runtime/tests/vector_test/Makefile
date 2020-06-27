
RISCV_TOOLCHAIN_PATH ?= $(wildcard ../../../../riscv-gnu-toolchain/drops)

COMP = $(RISCV_TOOLCHAIN_PATH)/bin/riscv32-unknown-elf-gcc

CC_FLAGS = -ffreestanding -O0 -Wl,--gc-sections -nostartfiles -nostdlib -nostartfiles -nodefaultlibs -Wl,-Bstatic,-T,../../startup/vx_link.ld -march=rv32imv -mabi=ilp32

DMP  = $(RISCV_TOOLCHAIN_PATH)/bin/riscv32-unknown-elf-objdump
CPY  = $(RISCV_TOOLCHAIN_PATH)/bin/riscv32-unknown-elf-objcopy


# VX_STR  = ../../startup/vx_start.S



NEWLIB  = ../../newlib/newlib.c
VX_STR  = ../../startup/vx_start.S
VX_INT  = ../../intrinsics/vx_intrinsics.s
VX_IO   = ../../io/vx_io.S ../../io/vx_io.c
VX_API  = ../../vx_api/vx_api.c
VX_FIO  = ../../fileio/fileio.S
VX_VEC  = vx_vec.s
LIBS    = $(RISCV_TOOLCHAIN_PATH)/riscv32-unknown-elf/lib/libc.a $(RISCV_TOOLCHAIN_PATH)/riscv32-unknown-elf/lib/libstdc++.a -static-libgcc -lgcc

VX_MAIN = vx_vector_main

all: HEX DUMP ELF

DUMP: ELF
	$(DMP) -D $(VX_MAIN).elf > $(VX_MAIN).dump

HEX: ELF
	$(CPY) -O ihex $(VX_MAIN).elf $(VX_MAIN).hex

ELF: 
	$(COMP) $(CC_FLAGS) $(VX_STR) $(VX_VEC) $(VX_FIO) $(NEWLIB) $(VX_INT) $(VX_IO) $(VX_API) $(VX_MAIN).c $(LIBS) -Iinclude  -o $(VX_MAIN).elf
