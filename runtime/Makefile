RISCV_TOOLCHAIN_PATH ?= /opt/riscv-gnu-toolchain

CC = $(RISCV_TOOLCHAIN_PATH)/bin/riscv32-unknown-elf-gcc
AR = $(RISCV_TOOLCHAIN_PATH)/bin/riscv32-unknown-elf-gcc-ar
DP = $(RISCV_TOOLCHAIN_PATH)/bin/riscv32-unknown-elf-objdump
CP = $(RISCV_TOOLCHAIN_PATH)/bin/riscv32-unknown-elf-objcopy

CFLAGS += -O3 -march=rv32imf -mabi=ilp32f -Wstack-usage=1024 -fno-exceptions -fdata-sections -ffunction-sections
CFLAGS += -I./include -I../hw

PROJECT = libvortexrt

SRCS = ./src/vx_start.S ./src/vx_syscalls.c ./src/vx_print.S ./src/vx_print.c ./src/vx_spawn.c ./src/vx_spawn.S ./src/vx_perf.c

OBJS := $(addsuffix .o, $(notdir $(SRCS)))

all: $(PROJECT).a $(PROJECT).dump

$(PROJECT).dump: $(PROJECT).a
	$(DP) -D $(PROJECT).a > $(PROJECT).dump

%.S.o: src/%.S
	$(CC) $(CFLAGS) -c $< -o $@

%.c.o: src/%.c
	$(CC) $(CFLAGS) -c $< -o $@

$(PROJECT).a: $(OBJS)
	$(AR) rcs $(PROJECT).a $^

.depend: $(SRCS)
	$(CC) $(CFLAGS) -MM $^ > .depend;

clean:
	rm -rf *.a *.o *.dump .depend 