ROOT_DIR := $(realpath ../../..)

ifeq ($(XLEN),64)
CFLAGS += -march=rv64imafd -mabi=lp64d
else
CFLAGS += -march=rv32imaf -mabi=ilp32f
endif
STARTUP_ADDR ?= 0x80000000

VORTEX_KN_PATH ?= $(ROOT_DIR)/sw/kernel

LLVM_CFLAGS += --sysroot=$(RISCV_SYSROOT)
LLVM_CFLAGS += --gcc-toolchain=$(RISCV_TOOLCHAIN_PATH)
LLVM_CFLAGS += -Xclang -target-feature -Xclang +vortex

CC  = $(LLVM_VORTEX)/bin/clang $(LLVM_CFLAGS)
CXX = $(LLVM_VORTEX)/bin/clang++ $(LLVM_CFLAGS)
AR  = $(LLVM_VORTEX)/bin/llvm-ar
DP  = $(LLVM_VORTEX)/bin/llvm-objdump
CP  = $(LLVM_VORTEX)/bin/llvm-objcopy

CFLAGS += -Wall -Wextra -Wfatal-errors -Werror -Wno-unused-command-line-argument
CFLAGS += -O3 -mcmodel=medany -fno-exceptions -nostartfiles -nostdlib -fdata-sections -ffunction-sections
CFLAGS += -I$(VORTEX_HOME)/sw/kernel/include -I$(ROOT_DIR)/sw -I$(ROOT_DIR)/hw -I$(SW_COMMON_DIR)
CFLAGS += -DXLEN_$(XLEN) -DNDEBUG $(CONFIGS) -D__VORTEX__

LIBC_LIB += -L$(LIBC_VORTEX)/lib -lm -lc
LIBC_LIB += $(LIBCRT_VORTEX)/lib/baremetal/libclang_rt.builtins-riscv$(XLEN).a

LDFLAGS += -Wl,-Bstatic,--gc-sections,-T,$(VORTEX_HOME)/sw/kernel/scripts/link$(XLEN).ld,--defsym=STARTUP_ADDR=$(STARTUP_ADDR) $(VORTEX_KN_PATH)/libvortex.a $(LIBC_LIB)

VX_STARTUP_SRC := $(VORTEX_HOME)/sw/kernel/src/vx_start.S
APP_OBJS = $(addsuffix .o, $(basename $(notdir $(SRCS))))
KERNEL_STARTUP := $(VORTEX_HOME)/sw/kernel/scripts/kernel_startup.sh

all: $(PROJECT).elf $(PROJECT).vxbin $(PROJECT).dump

$(PROJECT).dump: $(PROJECT).elf
	$(DP) -D $< > $@

$(PROJECT).vxbin: $(PROJECT).elf
	OBJCOPY=$(CP) $(VORTEX_HOME)/sw/kernel/scripts/vxbin.py $< $@

$(VORTEX_KN_PATH)/libvortex.a:
	$(MAKE) -C $(VORTEX_KN_PATH)

vx_start.o: $(SRCS) $(VORTEX_KN_PATH)/libvortex.a
	$(CC) $(CFLAGS) -c $(SRCS)
	$(CC) $(CFLAGS) -DNEED_GP -DNEED_TLS -DNEED_INITFINI -c $(VX_STARTUP_SRC) -o $@
	$(CC) $(CFLAGS) $@ $(APP_OBJS) $(LDFLAGS) -o $@.elf
	$(CC) $(CFLAGS) $$($(KERNEL_STARTUP) $(DP) $@.elf) -c $(VX_STARTUP_SRC) -o $@ && rm -f $@.elf

$(PROJECT).elf: vx_start.o $(SRCS) $(VORTEX_KN_PATH)/libvortex.a
	$(CC) $(CFLAGS) vx_start.o $(APP_OBJS) $(LDFLAGS) -o $@

run-rtlsim: $(PROJECT).vxbin
	$(ROOT_DIR)/sim/rtlsim/rtlsim $(PROJECT).vxbin

run-simx: $(PROJECT).vxbin
	$(ROOT_DIR)/sim/simx/simx $(PROJECT).vxbin

.depend: $(SRCS)
	$(CC) $(CFLAGS) -MM $^ > .depend;

clean:
	rm -rf *.elf *.bin *.dump *.o *.log .depend
