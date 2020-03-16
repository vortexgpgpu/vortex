RISCV_TOOL_PATH ?= $(wildcard ~/dev/riscv-gnu-toolchain/drops)
VX_RT_PATH ?= $(wildcard ../../runtime)

CC   = $(RISCV_TOOL_PATH)/bin/riscv32-unknown-elf-gcc
CXX  = $(RISCV_TOOL_PATH)/bin/riscv32-unknown-elf-g++
DMP  = $(RISCV_TOOL_PATH)/bin/riscv32-unknown-elf-objdump
HEX  = $(RISCV_TOOL_PATH)/bin/riscv32-unknown-elf-objcopy
GDB  = $(RISCV_TOOL_PATH)/bin/riscv32-unknown-elf-gdb

NEWLIB  = $(VX_RT_PATH)/newlib/newlib.c
VX_STR  = $(VX_RT_PATH)/startup/vx_start.s
VX_INT  = $(VX_RT_PATH)/intrinsics/vx_intrinsics.s
VX_IO   = $(VX_RT_PATH)/io/vx_io.s $(VX_RT_PATH)/io/vx_io.c
VX_API  = $(VX_RT_PATH)/vx_api/vx_api.c
VX_FIO  = $(VX_RT_PATH)/fileio/fileio.s

CFLAGS = -march=rv32im -mabi=ilp32 -O0 -Wl,-Bstatic,-T,$(VX_RT_PATH)/mains/vortex_link.ld -ffreestanding -nostdlib

LIBS    = $(RISCV_TOOL_PATH)/riscv32-unknown-elf/lib/libc.a $(RISCV_TOOL_PATH)/riscv32-unknown-elf/lib/libstdc++.a -static-libgcc -lgcc

PROJECT = demo

SRCS = main.c

all: $(PROJECT).dump $(PROJECT).hex

$(PROJECT).dump: $(PROJECT).elf
	$(DMP) -D $(PROJECT).elf > $(PROJECT).dump

$(PROJECT).hex: $(PROJECT).elf
	$(HEX) -O ihex $(PROJECT).elf $(PROJECT).hex

$(PROJECT).elf: $(SRCS)
	$(CC) $(CFLAGS) $(VX_STR) $(VX_FIO) $(NEWLIB) $(VX_INT) $(VX_IO) $(VX_API) $(SRCS) $(LIBS) -I$(VX_RT_PATH)  -o $(PROJECT).elf

clean:
	rm -rf *.o *.elf *.dump *.hex *.qemu *.log *.debug
