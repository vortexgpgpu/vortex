ROOT_DIR := $(realpath ../../../../..)
include $(ROOT_DIR)/config.mk

ifeq ($(XLEN),64)
CFLAGS += -march=rv64imafd -mabi=lp64d
else
CFLAGS += -march=rv32imaf -mabi=ilp32f
endif

SRC_DIR := $(VORTEX_HOME)/hw/syn/xilinx/test/kernel

SCRIPT_DIR := $(VORTEX_HOME)/hw/scripts

BIN2COE_PATH ?= $(SCRIPT_DIR)/bin2coe

CC = $(RISCV_TOOLCHAIN_PATH)/bin/$(RISCV_PREFIX)-gcc
AR = $(RISCV_TOOLCHAIN_PATH)/bin/$(RISCV_PREFIX)-gcc-ar
DP = $(RISCV_TOOLCHAIN_PATH)/bin/$(RISCV_PREFIX)-objdump
CP = $(RISCV_TOOLCHAIN_PATH)/bin/$(RISCV_PREFIX)-objcopy

CFLAGS += -O3 -Wstack-usage=1024 -ffreestanding -nostartfiles -fdata-sections -ffunction-sections
CFLAGS += -I$(VORTEX_RT_PATH)/include -I$(VORTEX_HOME)/hw

LDFLAGS += -lm -Wl,-Bstatic,-T,$(VORTEX_RT_PATH)/scripts/link$(XLEN).ld,--defsym=STARTUP_ADDR=0x80000000

PROJECT = kernel

SRCS = $(SRC_DIR)/main.c $(SRC_DIR)/start.S

all: $(PROJECT).elf $(PROJECT).hex $(PROJECT).bin $(PROJECT).dump $(PROJECT).bin.coe

$(PROJECT).dump: $(PROJECT).elf
	$(DP) -D $< > $@

$(PROJECT).hex: $(PROJECT).elf
	$(CP) -O ihex $< $@

$(PROJECT).bin: $(PROJECT).elf
	$(CP) -O binary $< $@

$(PROJECT).bin.coe: $(PROJECT).bin
	$(BIN2COE_PATH)/bin2coe $< --out=$@ --binary=$(PROJECT).bin --data=$(PROJECT).dat --binaddr=8192 --depth=16384 --wordsize=64

$(PROJECT).elf: $(SRCS)
	$(CC) $(CFLAGS) $^ $(LDFLAGS) -o $@

.depend: $(SRCS)
	$(CC) $(CFLAGS) -MM $^ > .depend;

clean:
	rm -rf *.bin *.elf *.hex *.dump *.coe .depend 
