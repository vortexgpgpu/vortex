XLEN ?= 32

ifeq ($(XLEN),32)
RISCV_TOOLCHAIN_PATH ?= /opt/riscv-gnu-toolchain
else
RISCV_TOOLCHAIN_PATH ?= /opt/riscv64-gnu-toolchain
endif

RISCV_PREFIX ?= riscv$(XLEN)-unknown-elf-

LLVM_VORTEX ?= /opt/llvm-vortex

LLVM_CFLAGS += --sysroot=$(RISCV_TOOLCHAIN_PATH)/riscv32-unknown-elf
LLVM_CFLAGS += --gcc-toolchain=$(RISCV_TOOLCHAIN_PATH)
LLVM_CFLAGS += -Xclang -target-feature -Xclang +vortex -mllvm -vortex-branch-divergence=0
#LLVM_CFLAGS += -I$(RISCV_TOOLCHAIN_PATH)/riscv32-unknown-elf/include/c++/9.2.0/riscv32-unknown-elf 
#LLVM_CFLAGS += -I$(RISCV_TOOLCHAIN_PATH)/riscv32-unknown-elf/include/c++/9.2.0
#LLVM_CFLAGS += -Wl,-L$(RISCV_TOOLCHAIN_PATH)/lib/gcc/riscv32-unknown-elf/9.2.0
#LLVM_CFLAGS += --rtlib=libgcc

CC  = $(LLVM_VORTEX)/bin/clang $(LLVM_CFLAGS)
CXX = $(LLVM_VORTEX)/bin/clang++ $(LLVM_CFLAGS)
DP  = $(LLVM_VORTEX)/bin/llvm-objdump
CP  = $(LLVM_VORTEX)/bin/llvm-objcopy

#CC = $(RISCV_TOOLCHAIN_PATH)/bin/$(RISCV_PREFIX)gcc
#AR = $(RISCV_TOOLCHAIN_PATH)/bin/$(RISCV_PREFIX)gcc-ar
#DP = $(RISCV_TOOLCHAIN_PATH)/bin/$(RISCV_PREFIX)objdump
#CP = $(RISCV_TOOLCHAIN_PATH)/bin/$(RISCV_PREFIX)objcopy

ifeq ($(XLEN),32)
CFLAGS += -march=rv32imf -mabi=ilp32f
else
CFLAGS += -march=rv64imfd -mabi=lp64d
endif

CFLAGS += -O3 -mcmodel=medany -fno-exceptions -fdata-sections -ffunction-sections
CFLAGS += -I./include -I../hw

PROJECT = libvortexrt

SRCS = ./src/vx_start.S ./src/vx_syscalls.c ./src/vx_print.S ./src/tinyprintf.c ./src/vx_print.c ./src/vx_spawn.c ./src/vx_spawn.S ./src/vx_perf.c

OBJS := $(addsuffix .o, $(notdir $(SRCS)))

all: $(PROJECT).a $(PROJECT).dump

$(PROJECT).dump: $(PROJECT).a
	$(DP) -D $(PROJECT).a > $(PROJECT).dump

%.S.o: src/%.S
	$(CC) $(CFLAGS) -c $< -o $@

%.c.o: src/%.c
	$(CC) $(CFLAGS) -c $< -o $@

$(PROJECT).a: $(OBJS)
	$(AR) rcs $@ $^

.depend: $(SRCS)
	$(CC) $(CFLAGS) -MM $^ > .depend;

clean:
	rm -rf *.a *.o *.dump .depend 