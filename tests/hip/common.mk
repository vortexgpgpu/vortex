# Shared build rules for tests/hip/*: build the host binary via chipStar's
# hipcc (HIP -> SPIR-V), build the Vortex runtime driver, then run under
# POCL/chipStar which JIT-compiles the SPIR-V to Vortex.
#
# chipStar HIP is currently rv64-only: hipcc emits Physical64 SPIR-V
# unconditionally, and POCL refuses Physical64 on a 32-bit Vortex device.
# The skip is enforced at the regression-script level (ci/regression.sh).

CHIPSTAR_PATH ?= $(TOOLDIR)/chipstar
POCL_PATH     ?= $(TOOLDIR)/pocl
HIPCC         ?= $(CHIPSTAR_PATH)/bin/hipcc
HIP_CLANG_PATH ?= $(LLVM_PATH)/bin

OPTS ?=

VORTEX_RT_SRC ?= $(ROOT_DIR)/sw/runtime
VORTEX_RT_LIB ?= $(VORTEX_RT_SRC)
VORTEX_KN_PATH ?= $(ROOT_DIR)/sw/kernel

# FPGA/HW backend selectors — same names + defaults as tests/regression.
#   opae : TARGET picks the libopae shape (real FPGA vs ASE/opaesim emulators)
#   xrt  : TARGET picks among Xilinx hw / hw_emu / sw_emu;
#          FPGA_BIN_DIR points at the built xclbin tree for hw/hw_emu.
TARGET ?= opaesim
XRT_SYN_DIR ?= $(VORTEX_HOME)/hw/syn/xilinx/xrt
XRT_DEVICE_INDEX ?= 0

ifeq ($(TARGET), fpga)
    OPAE_DRV_PATHS ?= libopae-c.so
else ifeq ($(TARGET), asesim)
    OPAE_DRV_PATHS ?= libopae-c-ase.so
else ifeq ($(TARGET), opaesim)
    OPAE_DRV_PATHS ?= libopae-c-sim.so
endif

# Device-side flags POCL re-passes to clang when JIT'ing the SPIR-V
# coming out of chipStar. Same shape as tests/opencl/common.mk.
VX_LIBS += -L$(LIBC_PATH)/lib -lm -lc
VX_LIBS += $(LIBCRT_PATH)/lib/baremetal/libclang_rt.builtins-riscv$(XLEN).a

VX_CFLAGS  += --target=riscv$(XLEN)-unknown-elf
VX_CFLAGS  += -O3 -mcmodel=medany --sysroot=$(RISCV_SYSROOT) --gcc-toolchain=$(RISCV_TOOLCHAIN_PATH)
VX_CFLAGS  += -fno-rtti -fno-exceptions -nostartfiles -nostdlib -fdata-sections -ffunction-sections
VX_CFLAGS  += -I$(ROOT_DIR)/sw -I$(ROOT_DIR)/hw -I$(VORTEX_HOME)/sw/kernel/include
VX_CFLAGS  += -DVX_CFG_XLEN=$(XLEN) -DVX_CFG_XLEN_$(XLEN) -DNDEBUG -D__VORTEX__
VX_CFLAGS  += -Xclang -target-feature -Xclang +xvortex
VX_CFLAGS  += -Xclang -target-feature -Xclang +zicond
VX_CFLAGS  += -mllvm -disable-loop-idiom-all

ifeq ($(XLEN),64)
	VX_CFLAGS += -march=rv64imafd -mabi=lp64d
else
	VX_CFLAGS += -march=rv32imaf -mabi=ilp32f
endif

VX_LDFLAGS += -fuse-ld=lld -Wl,-z,norelro
VX_LDFLAGS += -Wl,-Bstatic,--gc-sections,-T$(VORTEX_HOME)/sw/kernel/scripts/link$(XLEN).ld,--defsym=STARTUP_ADDR=$(STARTUP_ADDR)
VX_LDFLAGS += $(VORTEX_KN_PATH)/libvortex2.a $(VX_LIBS)

VX_BINTOOL += OBJCOPY=$(LLVM_PATH)/bin/llvm-objcopy $(VORTEX_HOME)/sw/kernel/scripts/vxbin.py

# chipStar emits -cl-std=CL3.0; Vortex reports CL 1.2. POCL would refuse
# with CL_BUILD_PROGRAM_FAILURE without IGNORE_CL_STD. Safe for SPIR-V
# input (kernels are pre-compiled, OpenCL C version doesn't apply).
POCL_CC_FLAGS += POCL_IGNORE_CL_STD=1

# chipStar links against the system OpenCL ICD loader (libOpenCL.so.1).
# PoCL is built ICD-only and ships a vendor .icd, so the loader discovers
# the Vortex platform directly via OCL_ICD_VENDORS -- no LD_PRELOAD shim is
# needed (this replaces the old redirect to POCL's standalone libOpenCL.so).
# OCL_ICD_LIB_DIR pins the ocl-icd loader ahead of any other vendor loader
# present on the host (e.g. CUDA or Xilinx XRT).
OCL_ICD_VENDORS ?= $(POCL_PATH)/etc/OpenCL/vendors
OCL_ICD_LIB_DIR ?= /usr/lib/x86_64-linux-gnu
HIP_OCL_ENV = OCL_ICD_VENDORS=$(OCL_ICD_VENDORS)
POCL_CC_FLAGS += POCL_VORTEX_XLEN=$(XLEN) LLVM_PREFIX=$(LLVM_PATH)
POCL_CC_FLAGS += POCL_VORTEX_BINTOOL="$(VX_BINTOOL)"
POCL_CC_FLAGS += POCL_VORTEX_CFLAGS="$(VX_CFLAGS)"
POCL_CC_FLAGS += POCL_VORTEX_LDFLAGS="$(VX_LDFLAGS)"

HIPCC_FLAGS += -std=c++17
HIPCC_FLAGS += $(CONFIGS)
# The embedded .hipInfo OFFLOAD_TRIPLE selects the primary SPIR-V pointer
# width. hipcc rewrites it to match XLEN so the emitted SPIR-V is
# acceptable to POCL Vortex (POCL refuses Physical32 on rv64 and vice-versa).
HIPCC_FLAGS += --offload-pointer-width=$(XLEN)

# Stock clang on Ubuntu 22.04 auto-selects the highest-numbered gcc
# (currently 12), but the default Ubuntu package set only ships the
# libstdc++ headers (cstddef, etc.) under gcc-11. Force the gcc-11 dir
# when its headers are present.
ifneq (,$(wildcard /usr/include/c++/11/cstddef))
HIPCC_FLAGS += --gcc-install-dir=/usr/lib/gcc/x86_64-linux-gnu/11
endif

# Point hipcc's HIP install root at the (possibly relocated) chipStar.
# With HIPCC-patches/0002, hipcc self-locates and rewrites the baked
# build-time prefix in its embedded offload flags to wherever it actually
# lives, so --hip-path is belt-and-suspenders (and keeps the host link
# pointed at the relocated libCHIP.so via the -L below).
HIPCC_FLAGS += --hip-path=$(CHIPSTAR_PATH)
HIPCC_FLAGS += -L$(CHIPSTAR_PATH)/lib -Wl,-rpath,$(CHIPSTAR_PATH)/lib

# libOpenCL.so (POCL with Vortex device driver linked in) needs
# libvortex.so to resolve at link time. Pass -rpath-link to the linker
# so it can chase the transitive dep without us linking vortex directly.
HIPCC_FLAGS += -Wl,-rpath-link,$(VORTEX_RT_LIB)
HIPCC_FLAGS += -Wl,-rpath-link,$(POCL_PATH)/lib
HIPCC_FLAGS += -Wl,-rpath-link,$(LLVM_PATH)/lib

ifdef DEBUG
	HIPCC_FLAGS   += -g -O0
	POCL_CC_FLAGS += POCL_DEBUG=all
else
	HIPCC_FLAGS   += -O2
endif

STARTUP_ADDR ?= 0x80000000

all: $(PROJECT)

RUNTIME_ARGS = CONFIGS="$(CONFIGS)" $(if $(DEBUG),DEBUG=$(DEBUG)) $(if $(PERF),PERF=$(PERF))

$(VORTEX_KN_PATH)/libvortex2.a:
	$(MAKE) -C $(VORTEX_KN_PATH)

$(VORTEX_RT_LIB)/libvortex.so:
	$(RUNTIME_ARGS) $(MAKE) -C $(VORTEX_RT_SRC)/stub DESTDIR=$(VORTEX_RT_LIB)

# chipStar's hipcc handles host+device in one invocation, producing a
# host ELF that embeds the SPIR-V module. The build dispatches to
# llvm-spirv (translator) which needs LD_LIBRARY_PATH to find LLVM .so
# at build time -- POCL_CC_FLAGS env is only used at run time.
$(PROJECT): $(SRCS) common.h $(VORTEX_KN_PATH)/libvortex2.a $(VORTEX_RT_LIB)/libvortex.so
	HIP_CLANG_PATH=$(HIP_CLANG_PATH) LD_LIBRARY_PATH=$(LLVM_PATH)/lib:$(LD_LIBRARY_PATH) $(HIPCC) $(HIPCC_FLAGS) -I. $< -o $@

run-simx: $(PROJECT)
	$(RUNTIME_ARGS) $(MAKE) -C $(VORTEX_RT_SRC)/simx DESTDIR=$(VORTEX_RT_LIB)
	$(HIP_OCL_ENV) LD_LIBRARY_PATH=$(OCL_ICD_LIB_DIR):$(CHIPSTAR_PATH)/lib:$(POCL_PATH)/lib:$(VORTEX_RT_LIB):$(LLVM_PATH)/lib:$(LD_LIBRARY_PATH) $(POCL_CC_FLAGS) VORTEX_DRIVER=simx ./$(PROJECT) $(OPTS)

run-rtlsim: $(PROJECT)
	$(RUNTIME_ARGS) $(MAKE) -C $(VORTEX_RT_SRC)/rtlsim DESTDIR=$(VORTEX_RT_LIB)
	$(HIP_OCL_ENV) LD_LIBRARY_PATH=$(OCL_ICD_LIB_DIR):$(CHIPSTAR_PATH)/lib:$(POCL_PATH)/lib:$(VORTEX_RT_LIB):$(LLVM_PATH)/lib:$(LD_LIBRARY_PATH) $(POCL_CC_FLAGS) VORTEX_DRIVER=rtlsim ./$(PROJECT) $(OPTS)

run-opae: $(PROJECT)
	$(RUNTIME_ARGS) $(MAKE) -C $(VORTEX_RT_SRC)/opae DESTDIR=$(VORTEX_RT_LIB)
	$(HIP_OCL_ENV) SCOPE_JSON_PATH=$(VORTEX_RT_LIB)/scope.json OPAE_DRV_PATHS=$(OPAE_DRV_PATHS) LD_LIBRARY_PATH=$(OCL_ICD_LIB_DIR):$(CHIPSTAR_PATH)/lib:$(POCL_PATH)/lib:$(VORTEX_RT_LIB):$(LLVM_PATH)/lib:$(LD_LIBRARY_PATH) $(POCL_CC_FLAGS) VORTEX_DRIVER=opae ./$(PROJECT) $(OPTS)

run-xrt: $(PROJECT)
	$(RUNTIME_ARGS) $(MAKE) -C $(VORTEX_RT_SRC)/xrt DESTDIR=$(VORTEX_RT_LIB)
ifeq ($(TARGET), hw)
	$(HIP_OCL_ENV) SCOPE_JSON_PATH=$(FPGA_BIN_DIR)/scope.json XRT_INI_PATH=$(VORTEX_RT_SRC)/xrt/xrt.ini EMCONFIG_PATH=$(FPGA_BIN_DIR) XRT_DEVICE_INDEX=$(XRT_DEVICE_INDEX) XRT_XCLBIN_PATH=$(FPGA_BIN_DIR)/vortex_afu.xclbin LD_LIBRARY_PATH=$(XILINX_XRT)/lib:$(CHIPSTAR_PATH)/lib:$(POCL_PATH)/lib:$(VORTEX_RT_LIB):$(LLVM_PATH)/lib:$(LD_LIBRARY_PATH) $(POCL_CC_FLAGS) VORTEX_DRIVER=xrt ./$(PROJECT) $(OPTS)
else ifeq ($(TARGET), hw_emu)
	$(HIP_OCL_ENV) SCOPE_JSON_PATH=$(FPGA_BIN_DIR)/scope.json XCL_EMULATION_MODE=$(TARGET) XRT_INI_PATH=$(VORTEX_RT_SRC)/xrt/xrt.ini EMCONFIG_PATH=$(FPGA_BIN_DIR) XRT_DEVICE_INDEX=$(XRT_DEVICE_INDEX) XRT_XCLBIN_PATH=$(FPGA_BIN_DIR)/vortex_afu.xclbin LD_LIBRARY_PATH=$(XILINX_XRT)/lib:$(CHIPSTAR_PATH)/lib:$(POCL_PATH)/lib:$(VORTEX_RT_LIB):$(LLVM_PATH)/lib:$(LD_LIBRARY_PATH) $(POCL_CC_FLAGS) VORTEX_DRIVER=xrt ./$(PROJECT) $(OPTS)
else
	$(HIP_OCL_ENV) SCOPE_JSON_PATH=$(VORTEX_RT_LIB)/scope.json LD_LIBRARY_PATH=$(XILINX_XRT)/lib:$(CHIPSTAR_PATH)/lib:$(POCL_PATH)/lib:$(VORTEX_RT_LIB):$(LLVM_PATH)/lib:$(LD_LIBRARY_PATH) $(POCL_CC_FLAGS) VORTEX_DRIVER=xrt ./$(PROJECT) $(OPTS)
endif

clean:
	rm -f $(PROJECT) *.o *.vxbin *.dump *.ll *.log *.spv common.h

.PHONY: all run-simx run-rtlsim run-opae run-xrt clean
