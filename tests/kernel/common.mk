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

VORTEX_KN_PATH ?= $(realpath ../../../kernel)

CC = $(RISCV_TOOLCHAIN_PATH)/bin/$(RISCV_PREFIX)-gcc
AR = $(RISCV_TOOLCHAIN_PATH)/bin/$(RISCV_PREFIX)-gcc-ar
DP = $(RISCV_TOOLCHAIN_PATH)/bin/$(RISCV_PREFIX)-objdump
CP = $(RISCV_TOOLCHAIN_PATH)/bin/$(RISCV_PREFIX)-objcopy

SIM_DIR = ../../../sim

CFLAGS += -O3 -mcmodel=medany -fno-exceptions -nostartfiles -fdata-sections -ffunction-sections
CFLAGS += -I$(VORTEX_KN_PATH)/include -I$(VORTEX_KN_PATH)/../hw

LDFLAGS += -lm -Wl,-Bstatic,--gc-sections,-T,$(VORTEX_KN_PATH)/linker/vx_link$(XLEN).ld,--defsym=STARTUP_ADDR=0x80000000 $(VORTEX_KN_PATH)/libvortexrt.a

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
	$(SIM_DIR)/simx/simx $(PROJECT).bin

.depend: $(SRCS)
	$(CC) $(CFLAGS) -MM $^ > .depend;

clean:
	rm -rf *.elf *.bin *.dump .depend 
