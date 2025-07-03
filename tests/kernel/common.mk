ROOT_DIR := $(realpath ../../..)

ifeq ($(XLEN),64)
CFLAGS += -march=rv64imafd -mabi=lp64d
else
CFLAGS += -march=rv32imaf -mabi=ilp32f
endif
STARTUP_ADDR ?= 0x80000000

VORTEX_KN_PATH ?= $(ROOT_DIR)/kernel

LLVM_CFLAGS += --sysroot=$(RISCV_SYSROOT)
LLVM_CFLAGS += --gcc-toolchain=$(RISCV_TOOLCHAIN_PATH)
LLVM_CFLAGS += -Xclang -target-feature -Xclang +vortex

CC  = $(LLVM_VORTEX)/bin/clang $(LLVM_CFLAGS)
CXX = $(LLVM_VORTEX)/bin/clang++ $(LLVM_CFLAGS)
AR  = $(LLVM_VORTEX)/bin/llvm-ar
DP  = $(LLVM_VORTEX)/bin/llvm-objdump
CP  = $(LLVM_VORTEX)/bin/llvm-objcopy

CFLAGS += -O3 -mcmodel=medany -fno-exceptions -nostartfiles -nostdlib -fdata-sections -ffunction-sections
CFLAGS += -I$(VORTEX_HOME)/kernel/include -I$(ROOT_DIR)/hw -I$(SW_COMMON_DIR)
CFLAGS += -DXLEN_$(XLEN) -DNDEBUG $(CONFIGS)

LIBC_LIB += -L$(LIBC_VORTEX)/lib -lm -lc
LIBC_LIB += $(LIBCRT_VORTEX)/lib/baremetal/libclang_rt.builtins-riscv$(XLEN).a

LDFLAGS += -Wl,-Bstatic,--gc-sections,-T,$(VORTEX_HOME)/kernel/scripts/link$(XLEN).ld,--defsym=STARTUP_ADDR=$(STARTUP_ADDR) $(VORTEX_KN_PATH)/libvortex.a $(LIBC_LIB)

all: $(PROJECT).elf $(PROJECT).bin $(PROJECT).dump

$(PROJECT).dump: $(PROJECT).elf
	$(DP) -D $< > $@

$(PROJECT).bin: $(PROJECT).elf
	$(CP) -O binary $< $@

$(PROJECT).elf: $(SRCS)
	$(CC) $(CFLAGS) $^ $(LDFLAGS) -o $@

run-rtlsim: $(PROJECT).bin
	$(ROOT_DIR)/sim/rtlsim/rtlsim $(PROJECT).bin

run-simx: $(PROJECT).bin
	$(ROOT_DIR)/sim/simx/simx $(PROJECT).bin

.depend: $(SRCS)
	$(CC) $(CFLAGS) -MM $^ > .depend;

clean:
	rm -rf *.elf *.bin *.dump *.log .depend
