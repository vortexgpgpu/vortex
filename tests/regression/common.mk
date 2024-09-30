ROOT_DIR := $(realpath ../../..)

TARGET ?= opaesim

XRT_SYN_DIR ?= $(VORTEX_HOME)/hw/syn/xilinx/xrt
XRT_DEVICE_INDEX ?= 0

VORTEX_RT_PATH ?= $(ROOT_DIR)/runtime
VORTEX_KN_PATH ?= $(ROOT_DIR)/kernel

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
VX_CFLAGS += -I$(VORTEX_HOME)/kernel/include -I$(ROOT_DIR)/hw
VX_CFLAGS += -DXLEN_$(XLEN)
VX_CFLAGS += -DNDEBUG

VX_LIBS += -L$(LIBC_VORTEX)/lib -lm -lc

VX_LIBS += $(LIBCRT_VORTEX)/lib/baremetal/libclang_rt.builtins-riscv$(XLEN).a
#VX_LIBS += -lgcc

VX_LDFLAGS += -Wl,-Bstatic,--gc-sections,-T,$(VORTEX_HOME)/kernel/scripts/link$(XLEN).ld,--defsym=STARTUP_ADDR=$(STARTUP_ADDR) $(VORTEX_KN_PATH)/libvortex.a $(VX_LIBS)

CXXFLAGS += -std=c++17 -Wall -Wextra -pedantic -Wfatal-errors
CXXFLAGS += -I$(VORTEX_HOME)/runtime/include -I$(ROOT_DIR)/hw

LDFLAGS += -L$(VORTEX_RT_PATH) -lvortex

# Debugging
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
	LD_LIBRARY_PATH=$(VORTEX_RT_PATH):$(LD_LIBRARY_PATH) VORTEX_DRIVER=simx ./$(PROJECT) $(OPTS)

run-rtlsim: $(PROJECT) kernel.vxbin
	LD_LIBRARY_PATH=$(VORTEX_RT_PATH):$(LD_LIBRARY_PATH) VORTEX_DRIVER=rtlsim ./$(PROJECT) $(OPTS)

run-opae: $(PROJECT) kernel.vxbin
	SCOPE_JSON_PATH=$(VORTEX_RT_PATH)/scope.json OPAE_DRV_PATHS=$(OPAE_DRV_PATHS) LD_LIBRARY_PATH=$(VORTEX_RT_PATH):$(LD_LIBRARY_PATH) VORTEX_DRIVER=opae ./$(PROJECT) $(OPTS)

run-xrt: $(PROJECT) kernel.vxbin
ifeq ($(TARGET), hw)
	SCOPE_JSON_PATH=$(FPGA_BIN_DIR)/scope.json XRT_INI_PATH=$(VORTEX_RT_PATH)/xrt/xrt.ini EMCONFIG_PATH=$(FPGA_BIN_DIR) XRT_DEVICE_INDEX=$(XRT_DEVICE_INDEX) XRT_XCLBIN_PATH=$(FPGA_BIN_DIR)/vortex_afu.xclbin LD_LIBRARY_PATH=$(XILINX_XRT)/lib:$(VORTEX_RT_PATH):$(LD_LIBRARY_PATH) VORTEX_DRIVER=xrt ./$(PROJECT) $(OPTS)
else ifeq ($(TARGET), hw_emu)
	SCOPE_JSON_PATH=$(FPGA_BIN_DIR)/scope.json XCL_EMULATION_MODE=$(TARGET) XRT_INI_PATH=$(VORTEX_RT_PATH)/xrt/xrt.ini EMCONFIG_PATH=$(FPGA_BIN_DIR) XRT_DEVICE_INDEX=$(XRT_DEVICE_INDEX) XRT_XCLBIN_PATH=$(FPGA_BIN_DIR)/vortex_afu.xclbin LD_LIBRARY_PATH=$(XILINX_XRT)/lib:$(VORTEX_RT_PATH):$(LD_LIBRARY_PATH) VORTEX_DRIVER=xrt ./$(PROJECT) $(OPTS)
else
	SCOPE_JSON_PATH=$(VORTEX_RT_PATH)/scope.json LD_LIBRARY_PATH=$(XILINX_XRT)/lib:$(VORTEX_RT_PATH):$(LD_LIBRARY_PATH) VORTEX_DRIVER=xrt ./$(PROJECT) $(OPTS)
endif

.depend: $(SRCS)
	$(CXX) $(CXXFLAGS) -MM $^ > .depend;

clean-kernel:
	rm -rf *.elf *.vxbin *.dump

clean-host:
	rm -rf $(PROJECT) *.o *.log .depend

clean: clean-kernel clean-host

ifneq ($(MAKECMDGOALS),clean)
    -include .depend
endif
