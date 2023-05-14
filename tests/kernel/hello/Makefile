XLEN ?= 32

ifeq ($(XLEN),64)
RISCV_TOOLCHAIN_PATH ?= /opt/riscv64-gnu-toolchain
else
RISCV_TOOLCHAIN_PATH ?= /opt/riscv-gnu-toolchain
endif

RISCV_PREFIX ?= riscv$(XLEN)-unknown-elf

VORTEX_KN_PATH ?= $(realpath ../../../kernel)

CC = $(RISCV_TOOLCHAIN_PATH)/bin/$(RISCV_PREFIX)-gcc
AR = $(RISCV_TOOLCHAIN_PATH)/bin/$(RISCV_PREFIX)-gcc-ar
DP = $(RISCV_TOOLCHAIN_PATH)/bin/$(RISCV_PREFIX)-objdump
CP = $(RISCV_TOOLCHAIN_PATH)/bin/$(RISCV_PREFIX)-objcopy

SIM_DIR=../../../sim

ifeq ($(XLEN),64)
CFLAGS += -march=rv64imafd -mabi=lp64d
else
CFLAGS += -march=rv32imaf -mabi=ilp32f
endif

CFLAGS += -O3 -v -mcmodel=medany -fno-rtti -fno-exceptions -nostartfiles -fdata-sections -ffunction-sections
CFLAGS += -I$(VORTEX_KN_PATH)/include -I$(VORTEX_KN_PATH)/../hw

LDFLAGS += -lm -Wl,-Bstatic,--gc-sections,-T,$(VORTEX_KN_PATH)/linker/vx_link$(XLEN).ld $(VORTEX_KN_PATH)/libvortexrt.a

PROJECT = hello

SRCS = main.cpp

all: $(PROJECT).elf $(PROJECT).bin $(PROJECT).dump

$(PROJECT).dump: $(PROJECT).elf
	$(DP) -D $(PROJECT).elf > $(PROJECT).dump

$(PROJECT).bin: $(PROJECT).elf
	$(CP) -O binary $(PROJECT).elf $(PROJECT).bin

$(PROJECT).elf: $(SRCS)
	$(CC) $(CFLAGS) $(SRCS) $(LDFLAGS) -o $(PROJECT).elf

run-rtlsim: $(PROJECT).bin
	$(SIM_DIR)/rtlsim/rtlsim $(PROJECT).bin

run-simx: $(PROJECT).bin
	$(SIM_DIR)/simx/simx -c 1 -i $(PROJECT).bin

.depend: $(SRCS)
	$(CC) $(CFLAGS) -MM $^ > .depend;

clean:
	rm -rf *.elf *.bin *.dump .depend 
