RISCV_TOOLCHAIN_PATH ?= /opt/riscv-gnu-toolchain
VORTEX_RT_PATH ?= $(wildcard ../..)

CC = $(RISCV_TOOLCHAIN_PATH)/bin/riscv32-unknown-elf-gcc
AR = $(RISCV_TOOLCHAIN_PATH)/bin/riscv32-unknown-elf-gcc-ar
DP = $(RISCV_TOOLCHAIN_PATH)/bin/riscv32-unknown-elf-objdump
CP = $(RISCV_TOOLCHAIN_PATH)/bin/riscv32-unknown-elf-objcopy

CFLAGS += -march=rv32imf -mabi=ilp32f -O3 -Wstack-usage=1024 -ffreestanding -nostartfiles -fdata-sections -ffunction-sections
CFLAGS += -I$(VORTEX_RT_PATH)/include -I$(VORTEX_RT_PATH)/../hw

LDFLAGS += -Wl,-Bstatic,-T,$(VORTEX_RT_PATH)/linker/vx_link.ld -Wl,--gc-sections $(VORTEX_RT_PATH)/libvortexrt.a

PROJECT = vx_dev_main

SRCS = vx_dev_main.c

all: $(PROJECT).elf $(PROJECT).hex $(PROJECT).dump

$(PROJECT).dump: $(PROJECT).elf
	$(DP) -D $(PROJECT).elf > $(PROJECT).dump

$(PROJECT).hex: $(PROJECT).elf
	$(CP) -O ihex $(PROJECT).elf $(PROJECT).hex

$(PROJECT).elf: $(SRCS)
	$(CC) $(CFLAGS) $(SRCS) $(LDFLAGS) -o $(PROJECT).elf

run: $(PROJECT).hex
	(cd ../../../hw/simulate/obj_dir && ./VVortex ../../../runtime/tests/dev/$(PROJECT).hex) 

.depend: $(SRCS)
	$(CC) $(CFLAGS) -MM $^ > .depend;

clean:
	rm -rf *.elf *.hex *.dump .depend 
