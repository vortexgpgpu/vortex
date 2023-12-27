XLEN ?= 32
TOOLDIR ?= /opt

ifeq ($(XLEN),64)
RISCV_TOOLCHAIN_PATH ?= $(TOOLDIR)/riscv64-gnu-toolchain
CFLAGS += -march=rv64imafd -mabi=lp64d
else
RISCV_TOOLCHAIN_PATH ?= $(TOOLDIR)/riscv-gnu-toolchain
CFLAGS += -march=rv32imaf -mabi=ilp32f
endif

RISCV_PREFIX ?= riscv$(XLEN)-unknown-elf

VORTEX_RT_PATH ?= $(realpath ../../../../../kernel)
BIN2COE_PATH ?= ../../../../../../bin2coe

CC = $(RISCV_TOOLCHAIN_PATH)/bin/$(RISCV_PREFIX)-gcc
AR = $(RISCV_TOOLCHAIN_PATH)/bin/$(RISCV_PREFIX)-gcc-ar
DP = $(RISCV_TOOLCHAIN_PATH)/bin/$(RISCV_PREFIX)-objdump
CP = $(RISCV_TOOLCHAIN_PATH)/bin/$(RISCV_PREFIX)-objcopy

CFLAGS += -O3 -Wstack-usage=1024 -ffreestanding -nostartfiles -fdata-sections -ffunction-sections
CFLAGS += -I$(VORTEX_RT_PATH)/include -I$(VORTEX_RT_PATH)/../hw

LDFLAGS += -lm -Wl,-Bstatic,-T,$(VORTEX_RT_PATH)/linker/vx_link$(XLEN).ld,--defsym=STARTUP_ADDR=0x80000000

PROJECT = kernel

SRCS = main.c start.S

all: $(PROJECT).elf $(PROJECT).hex $(PROJECT).bin $(PROJECT).dump $(PROJECT).bin.coe

$(PROJECT).dump: $(PROJECT).elf
	$(DP) -D $(PROJECT).elf > $(PROJECT).dump

$(PROJECT).hex: $(PROJECT).elf
	$(CP) -O ihex $(PROJECT).elf $(PROJECT).hex

$(PROJECT).bin: $(PROJECT).elf
	$(CP) -O binary $(PROJECT).elf $(PROJECT).bin

$(PROJECT).bin.coe: $(PROJECT).bin
	$(BIN2COE_PATH)/bin2coe $(PROJECT).bin --out=$(PROJECT).bin.coe --binary=$(PROJECT).bin --data=$(PROJECT).dat --binaddr=8192 --depth=16384 --wordsize=64

$(PROJECT).elf: $(SRCS)
	$(CC) $(CFLAGS) $(SRCS) $(LDFLAGS) -o $(PROJECT).elf

.depend: $(SRCS)
	$(CC) $(CFLAGS) -MM $^ > .depend;

clean:
	rm -rf *.bin *.elf *.hex *.dump *.coe .depend 
