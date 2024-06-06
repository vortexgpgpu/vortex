ROOT_DIR := $(realpath ../../..)

TARGET ?= opaesim

XRT_SYN_DIR ?= $(VORTEX_HOME)/hw/syn/xilinx/xrt
XRT_DEVICE_INDEX ?= 0

ifeq ($(XLEN),64)
VX_CFLAGS += -march=rv64imafd -mabi=lp64d
STARTUP_ADDR ?= 0x180000000
else
VX_CFLAGS += -march=rv32imaf -mabi=ilp32f
STARTUP_ADDR ?= 0x80000000
endif

LLVM_CFLAGS += --sysroot=$(RISCV_SYSROOT)
LLVM_CFLAGS += --gcc-toolchain=$(RISCV_TOOLCHAIN_PATH)
LLVM_CFLAGS += -Xclang -target-feature -Xclang +vortex
LLVM_CFLAGS += -Xclang -target-feature -Xclang +zicond
LLVM_CFLAGS += -mllvm -disable-loop-idiom-all # disable memset/memcpy loop idiom
#LLVM_CFLAGS += -mllvm -vortex-branch-divergence=0
#LLVM_CFLAGS += -mllvm -print-after-all
#LLVM_CFLAGS += -I$(RISCV_SYSROOT)/include/c++/9.2.0/$(RISCV_PREFIX)
#LLVM_CFLAGS += -I$(RISCV_SYSROOT)/include/c++/9.2.0
#LLVM_CFLAGS += -Wl,-L$(RISCV_TOOLCHAIN_PATH)/lib/gcc/$(RISCV_PREFIX)/9.2.0
#LLVM_CFLAGS += --rtlib=libgcc

VX_CC  = $(LLVM_VORTEX)/bin/clang $(LLVM_CFLAGS)
VX_CXX = $(LLVM_VORTEX)/bin/clang++ $(LLVM_CFLAGS)
VX_DP  = $(LLVM_VORTEX)/bin/llvm-objdump
VX_CP  = $(LLVM_VORTEX)/bin/llvm-objcopy

#VX_CC  = $(RISCV_TOOLCHAIN_PATH)/bin/$(RISCV_PREFIX)-gcc
#VX_CXX = $(RISCV_TOOLCHAIN_PATH)/bin/$(RISCV_PREFIX)-g++
#VX_DP  = $(RISCV_TOOLCHAIN_PATH)/bin/$(RISCV_PREFIX)-objdump
#VX_CP  = $(RISCV_TOOLCHAIN_PATH)/bin/$(RISCV_PREFIX)-objcopy

VX_CFLAGS += -O3 -mcmodel=medany -fno-rtti -fno-exceptions -nostartfiles -nostdlib -fdata-sections -ffunction-sections
VX_CFLAGS += -I$(VORTEX_KN_PATH)/include -I$(ROOT_DIR)/hw
VX_CFLAGS += -DXLEN_$(XLEN)
VX_CFLAGS += -DNDEBUG

VX_LIBS += -L$(LIBC_VORTEX)/lib -lm -lc

VX_LIBS += $(LIBCRT_VORTEX)/lib/baremetal/libclang_rt.builtins-riscv$(XLEN).a
#VX_LIBS += -lgcc

VX_LDFLAGS += -Wl,-Bstatic,--gc-sections,-T,$(VORTEX_KN_PATH)/scripts/link$(XLEN).ld,--defsym=STARTUP_ADDR=$(STARTUP_ADDR) $(ROOT_DIR)/kernel/libvortexrt.a $(VX_LIBS)

CXXFLAGS += -std=c++11 -Wall -Wextra -pedantic -Wfatal-errors
CXXFLAGS += -I$(VORTEX_RT_PATH)/include -I$(ROOT_DIR)/hw

LDFLAGS += -L$(ROOT_DIR)/runtime/stub -lvortex

# Debugigng
ifdef DEBUG
	CXXFLAGS += -g -O0
else
	CXXFLAGS += -O2 -DNDEBUG
endif

ifeq ($(TARGET), fpga)
	OPAE_DRV_PATHS ?= libopae-c.so
else
ifeq ($(TARGET), asesim)
	OPAE_DRV_PATHS ?= libopae-c-ase.so
else
ifeq ($(TARGET), opaesim)
	OPAE_DRV_PATHS ?= libopae-c-sim.so
endif
endif
endif

all: $(PROJECT) kernel.vxbin kernel.dump

kernel.dump: kernel.elf
	$(VX_DP) -D $< > $@

kernel.vxbin: kernel.elf
	OBJCOPY=$(VX_CP) $(VORTEX_HOME)/kernel/scripts/vxbin.py $< $@

kernel.elf: $(VX_SRCS)
	$(VX_CXX) $(VX_CFLAGS) $^ $(VX_LDFLAGS) -o kernel.elf

$(PROJECT): $(SRCS)
	$(CXX) $(CXXFLAGS) $^ $(LDFLAGS) -o $@

run-simx: $(PROJECT) kernel.vxbin
	LD_LIBRARY_PATH=$(ROOT_DIR)/runtime/simx:$(LD_LIBRARY_PATH) ./$(PROJECT) $(OPTS)

run-opae: $(PROJECT) kernel.vxbin
	SCOPE_JSON_PATH=$(ROOT_DIR)/runtime/opae/scope.json OPAE_DRV_PATHS=$(OPAE_DRV_PATHS) LD_LIBRARY_PATH=$(ROOT_DIR)/runtime/opae:$(LD_LIBRARY_PATH) ./$(PROJECT) $(OPTS)

run-rtlsim: $(PROJECT) kernel.vxbin
	LD_LIBRARY_PATH=$(ROOT_DIR)/runtime/rtlsim:$(LD_LIBRARY_PATH) ./$(PROJECT) $(OPTS)

run-xrt: $(PROJECT) kernel.vxbin
ifeq ($(TARGET), hw)
	XRT_INI_PATH=$(XRT_SYN_DIR)/xrt.ini EMCONFIG_PATH=$(FPGA_BIN_DIR) XRT_DEVICE_INDEX=$(XRT_DEVICE_INDEX) XRT_XCLBIN_PATH=$(FPGA_BIN_DIR)/vortex_afu.xclbin LD_LIBRARY_PATH=$(XILINX_XRT)/lib:$(ROOT_DIR)/runtime/xrt:$(LD_LIBRARY_PATH) ./$(PROJECT) $(OPTS)
else
	XCL_EMULATION_MODE=$(TARGET) XRT_INI_PATH=$(XRT_SYN_DIR)/xrt.ini EMCONFIG_PATH=$(FPGA_BIN_DIR) XRT_DEVICE_INDEX=$(XRT_DEVICE_INDEX) XRT_XCLBIN_PATH=$(FPGA_BIN_DIR)/vortex_afu.xclbin LD_LIBRARY_PATH=$(XILINX_XRT)/lib:$(ROOT_DIR)/runtime/xrt:$(LD_LIBRARY_PATH) ./$(PROJECT) $(OPTS)
endif

.depend: $(SRCS)
	$(CXX) $(CXXFLAGS) -MM $^ > .depend;

clean:
	rm -rf $(PROJECT) *.o *.log .depend

clean-all: clean
	rm -rf *.elf *.vxbin *.dump

ifneq ($(MAKECMDGOALS),clean)
    -include .depend
endif
