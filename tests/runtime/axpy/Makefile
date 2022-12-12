RISCV_TOOLCHAIN_PATH = /opt/riscv-vec

RISCV_PREFIX ?= riscv32-unknown-elf-
VORTEX_RT_PATH ?= $(realpath ../../../runtime)

CC = $(RISCV_TOOLCHAIN_PATH)/bin/$(RISCV_PREFIX)gcc
AR = $(RISCV_TOOLCHAIN_PATH)/bin/$(RISCV_PREFIX)gcc-ar
DP = $(RISCV_TOOLCHAIN_PATH)/bin/$(RISCV_PREFIX)objdump
CP = $(RISCV_TOOLCHAIN_PATH)/bin/$(RISCV_PREFIX)objcopy

SIM_DIR=../../../sim

CFLAGS += -march=rv32imfv -mabi=ilp32f -flax-vector-conversions -fno-inline-small-functions

CFLAGS += -O3 -Wstack-usage=1024 -mcmodel=medany -ffreestanding -nostartfiles -fdata-sections -ffunction-sections
CFLAGS += -I$(VORTEX_RT_PATH)/include -I$(VORTEX_RT_PATH)/../hw

LDFLAGS += -lm -Wl,-Bstatic,-T,$(VORTEX_RT_PATH)/linker/vx_link32.ld -Wl,--gc-sections $(VORTEX_RT_PATH)/libvortexrt.a

PROJECT = axpy

SRCS = main.cpp

all: $(PROJECT)_vector.elf $(PROJECT)_vector.bin $(PROJECT)_vector.dump

$(PROJECT)_vector.dump: $(PROJECT)_vector.elf
	$(DP) -D $(PROJECT)_vector.elf > $(PROJECT)_vector.dump

$(PROJECT)_vector.bin: $(PROJECT)_vector.elf
	$(CP) -O binary $(PROJECT)_vector.elf $(PROJECT)_vector.bin

$(PROJECT)_vector.elf: $(SRCS)
	$(CC) $(CFLAGS) -DUSE_RISCV_VECTOR $(SRCS) $(LDFLAGS) -o $(PROJECT)_vector.elf

run-simx-vector: $(PROJECT)_vector.bin
	$(SIM_DIR)/simx/simx -c 1 -i $(PROJECT)_vector.bin

.depend: $(SRCS)
	$(CC) $(CFLAGS) -MM $^ > .depend;

clean:
	rm -rf *.o *.elf *.bin *.dump .depend 
