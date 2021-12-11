RISCV_TOOLCHAIN_PATH ?= /opt/riscv-gnu-toolchain
RISCV64_TOOLCHAIN_PATH ?= /nethome/ssrivatsan8/riscv
VORTEX_RT_PATH ?= $(realpath ../../../runtime)

CC = $(RISCV64_TOOLCHAIN_PATH)/bin/riscv64-unknown-elf-gcc
AR = $(RISCV64_TOOLCHAIN_PATH)/bin/riscv64-unknown-elf-gcc-ar
DP = $(RISCV64_TOOLCHAIN_PATH)/bin/riscv64-unknown-elf-objdump
CP = $(RISCV64_TOOLCHAIN_PATH)/bin/riscv64-unknown-elf-objcopy

CFLAGS += -march=rv64imfd -mabi=lp64d -O3 -Wstack-usage=1024 -ffreestanding -nostartfiles -fdata-sections -ffunction-sections
CFLAGS += -I$(VORTEX_RT_PATH)/include -I$(VORTEX_RT_PATH)/../hw

LDFLAGS += -Wl,-Bstatic,-T,$(VORTEX_RT_PATH)/linker/vx_link64.ld -Wl,--gc-sections $(VORTEX_RT_PATH)/libvortexrt.a

PROJECT = simple64

SRCS = main.cpp

all: $(PROJECT).elf $(PROJECT).bin $(PROJECT).dump

$(PROJECT).dump: $(PROJECT).elf
	$(DP) -D $(PROJECT).elf > $(PROJECT).dump

$(PROJECT).bin: $(PROJECT).elf
	$(CP) -O binary $(PROJECT).elf $(PROJECT).bin

$(PROJECT).elf: $(SRCS)
	$(CC) $(CFLAGS) $(SRCS) $(LDFLAGS) -o $(PROJECT).elf

run-rtlsim: $(PROJECT).bin
	../../../sim/rtlsim/rtlsim $(PROJECT).bin

run-simx: $(PROJECT).bin
	../../../sim/simX/simX -a rv64i -c 1 -i $(PROJECT).bin

.depend: $(SRCS)
	$(CC) $(CFLAGS) -MM $^ > .depend;

clean:
	rm -rf *.elf *.bin *.dump .depend 
