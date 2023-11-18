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
RISCV_SYSROOT ?= $(RISCV_TOOLCHAIN_PATH)/$(RISCV_PREFIX)

LLVM_VORTEX ?= $(TOOLDIR)/llvm-vortex

LLVM_CFLAGS += --sysroot=$(RISCV_SYSROOT)
LLVM_CFLAGS += --gcc-toolchain=$(RISCV_TOOLCHAIN_PATH)
LLVM_CFLAGS += -Xclang -target-feature -Xclang +vortex -mllvm -vortex-branch-divergence=0
#LLVM_CFLAGS += -I$(RISCV_SYSROOT)/include/c++/9.2.0/$(RISCV_PREFIX) 
#LLVM_CFLAGS += -I$(RISCV_SYSROOT)/include/c++/9.2.0
#LLVM_CFLAGS += -Wl,-L$(RISCV_TOOLCHAIN_PATH)/lib/gcc/$(RISCV_PREFIX)/9.2.0
#LLVM_CFLAGS += --rtlib=libgcc

#CC  = $(LLVM_VORTEX)/bin/clang $(LLVM_CFLAGS)
#CXX = $(LLVM_VORTEX)/bin/clang++ $(LLVM_CFLAGS)
#DP  = $(LLVM_VORTEX)/bin/llvm-objdump
#CP  = $(LLVM_VORTEX)/bin/llvm-objcopy

CC  = $(RISCV_TOOLCHAIN_PATH)/bin/$(RISCV_PREFIX)-gcc
CXX = $(RISCV_TOOLCHAIN_PATH)/bin/$(RISCV_PREFIX)-g++
AR  = $(RISCV_TOOLCHAIN_PATH)/bin/$(RISCV_PREFIX)-gcc-ar
DP  = $(RISCV_TOOLCHAIN_PATH)/bin/$(RISCV_PREFIX)-objdump
CP  = $(RISCV_TOOLCHAIN_PATH)/bin/$(RISCV_PREFIX)-objcopy

CFLAGS += -O3 -mcmodel=medany -fno-exceptions -nostartfiles -fdata-sections -ffunction-sections
CFLAGS += -I./include -I../hw
CFLAGS += -DXLEN_$(XLEN)

PROJECT = libvortexrt

SRCS = ./src/vx_start.S ./src/vx_syscalls.c ./src/vx_print.S ./src/tinyprintf.c ./src/vx_print.c ./src/vx_spawn.c ./src/vx_serial.S ./src/vx_perf.c

OBJS := $(addsuffix .o, $(notdir $(SRCS)))

all: $(PROJECT).a $(PROJECT).dump

$(PROJECT).dump: $(PROJECT).a
	$(DP) -D $(PROJECT).a > $(PROJECT).dump

%.S.o: src/%.S
	$(CC) $(CFLAGS) -c $< -o $@

%.cpp.o: src/%.cpp
	$(CXX) $(CFLAGS) -c $< -o $@

%.c.o: src/%.c
	$(CC) $(CFLAGS) -c $< -o $@

$(PROJECT).a: $(OBJS)
	$(AR) rcs $@ $^

.depend: $(SRCS)
	$(CC) $(CFLAGS) -MM $^ > .depend;

clean:
	rm -rf *.a *.o *.dump .depend
