RISCV_TOOLCHAIN_PATH = /opt/riscv-vec

RISCV_PREFIX ?= riscv32-unknown-elf-
VORTEX_RT_PATH ?= $(realpath ../../../runtime)

CC = $(RISCV_TOOLCHAIN_PATH)/bin/$(RISCV_PREFIX)gcc
AR = $(RISCV_TOOLCHAIN_PATH)/bin/$(RISCV_PREFIX)gcc-ar
DP = $(RISCV_TOOLCHAIN_PATH)/bin/$(RISCV_PREFIX)objdump
CP = $(RISCV_TOOLCHAIN_PATH)/bin/$(RISCV_PREFIX)objcopy

SIM_DIR=../../../sim

CFLAGS += -march=rv32imfv -mabi=ilp32f -flax-vector-conversions -fpermissive

CFLAGS += -O3 -Wstack-usage=1024 -mcmodel=medany -ffreestanding -nostartfiles -fdata-sections -ffunction-sections
CFLAGS += -I$(VORTEX_RT_PATH)/include -I$(VORTEX_RT_PATH)/../hw

LDFLAGS += -lm -Wl,-Bstatic,-T,$(VORTEX_RT_PATH)/linker/vx_link32.ld -Wl,--gc-sections $(VORTEX_RT_PATH)/libvortexrt.a

PROJECT = pathfinder

SRCS = $(PROJECT).cpp

all: $(PROJECT)_serial.bin $(PROJECT)_vector.bin

$(PROJECT)_vector.dump: $(PROJECT)_vector.elf
	$(DP) -D $(PROJECT)_vector.elf > $(PROJECT)_vector.dump

$(PROJECT)_vector.bin: $(PROJECT)_vector.elf
	$(CP) -O binary $(PROJECT)_vector.elf $(PROJECT)_vector.bin

$(PROJECT)_vector.elf: $(SRCS)
	$(CC) $(CFLAGS) -DUSE_RISCV_VECTOR $(SRCS) $(LDFLAGS) -o $(PROJECT)_vector.elf -lstdc++

$(PROJECT)_serial.dump: $(PROJECT)_serial.elf
	$(DP) -D $(PROJECT)_serial.elf > $(PROJECT)_serial.dump

$(PROJECT)_serial.bin: $(PROJECT)_serial.elf
	$(CP) -O binary $(PROJECT)_serial.elf $(PROJECT)_serial.bin

$(PROJECT)_serial.elf: $(SRCS)
	$(CC) $(CFLAGS) $(SRCS) $(LDFLAGS) -o $(PROJECT)_serial.elf -lstdc++

run-simx-vector: $(PROJECT)_vector.bin
	$(SIM_DIR)/simx/simx -c 1 -i $(PROJECT)_vector.bin

run-simx-serial: $(PROJECT)_serial.bin
	$(SIM_DIR)/simx/simx -c 1 -i $(PROJECT)_serial.bin

.depend: $(SRCS)
	$(CC) $(CFLAGS) -MM $^ > .depend;

clean:
	rm -rf *.elf *.bin *.dump .depend 
