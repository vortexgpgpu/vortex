
RISCV_TOOLCHAIN_PATH ?= $(wildcard ../../../../riscv-gnu-toolchain/drops)

COMP = $(RISCV_TOOLCHAIN_PATH)/bin/riscv32-unknown-elf-gcc
# CC_FLAGS = -march=rv32im -mabi=ilp32 -O0 -Wl,-Bstatic,-T,linker.ld -ffreestanding -nostdlib
CC_FLAGS = -march=rv32im -mabi=ilp32 -O0 -Wl,-Bstatic,-T,../../startup/vx_link.ld -ffreestanding -nostartfiles

DMP  = $(RISCV_TOOLCHAIN_PATH)/bin/riscv32-unknown-elf-objdump
CPY  = $(RISCV_TOOLCHAIN_PATH)/bin/riscv32-unknown-elf-objcopy


VX_STR  = ../../startup/vx_start.S
VX_INT  = ../../intrinsics/vx_intrinsics.S
VX_IO   = ../../io/vx_io.S ../../io/vx_io.c
VX_API  = ../../vx_api/vx_api.c
VX_TEST = ../common/tests.c

VX_MAIN = ./vx_dev_main.c

all: HEX DUMP ELF

DUMP: ELF
	$(DMP) -D vx_dev_main.elf > vx_dev_main.dump

HEX: ELF
	$(CPY) -O ihex vx_dev_main.elf vx_dev_main.hex

ELF: 
	$(COMP) $(CC_FLAGS) $(VX_STR) $(VX_INT) $(VX_IO) $(VX_API) $(VX_TEST) $(VX_MAIN)  -o vx_dev_main.elf

clean:
	rm -rf *.o *.elf *.dump *.hex *.qemu *.log *.debug
