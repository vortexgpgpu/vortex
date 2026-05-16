# Shared build rules for tests/hip/*. Mirrors tests/opencl/common.mk:
# build the host binary via chipStar's hipcc (which lowers HIP -> SPIR-V),
# build the Vortex runtime driver, then run with the POCL/chipStar
# environment that lets POCL JIT the SPIR-V to Vortex.
#
# chipStar HIP is currently rv64-only: hipcc emits Physical64 SPIR-V
# unconditionally, and POCL refuses Physical64 on a 32-bit Vortex
# device. The skip is enforced at the regression-script level
# (ci/regression.sh, hip()).

CHIPSTAR_PATH ?= $(TOOLDIR)/chipstar
HIPCC         ?= $(CHIPSTAR_PATH)/bin/hipcc

OPTS ?=

VORTEX_RT_SRC ?= $(ROOT_DIR)/sw/runtime
VORTEX_RT_LIB ?= $(VORTEX_RT_SRC)
VORTEX_KN_PATH ?= $(ROOT_DIR)/sw/kernel

# Device-side flags POCL re-passes to clang when JIT'ing the SPIR-V
# coming out of chipStar. Same shape as tests/opencl/common.mk.
VX_LIBS += -L$(LIBC_VORTEX)/lib -lm -lc
VX_LIBS += $(LIBCRT_VORTEX)/lib/baremetal/libclang_rt.builtins-riscv$(XLEN).a

VX_CFLAGS  += --target=riscv$(XLEN)-unknown-elf
VX_CFLAGS  += -O3 -mcmodel=medany --sysroot=$(RISCV_SYSROOT) --gcc-toolchain=$(RISCV_TOOLCHAIN_PATH)
VX_CFLAGS  += -fno-rtti -fno-exceptions -nostartfiles -nostdlib -fdata-sections -ffunction-sections
VX_CFLAGS  += -I$(ROOT_DIR)/sw -I$(ROOT_DIR)/hw -I$(VORTEX_HOME)/sw/kernel/include
VX_CFLAGS  += -DXLEN_$(XLEN) -DNDEBUG -D__VORTEX__
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

VX_BINTOOL += OBJCOPY=$(LLVM_VORTEX)/bin/llvm-objcopy $(VORTEX_HOME)/sw/kernel/scripts/vxbin.py

# chipStar emits -cl-std=CL3.0; Vortex reports CL 1.2. POCL would refuse
# with CL_BUILD_PROGRAM_FAILURE without IGNORE_CL_STD. Safe for SPIR-V
# input (kernels are pre-compiled, OpenCL C version doesn't apply).
POCL_CC_FLAGS += POCL_IGNORE_CL_STD=1
POCL_CC_FLAGS += POCL_VORTEX_XLEN=$(XLEN) LLVM_PREFIX=$(LLVM_VORTEX)
POCL_CC_FLAGS += POCL_VORTEX_BINTOOL="$(VX_BINTOOL)"
POCL_CC_FLAGS += POCL_VORTEX_CFLAGS="$(VX_CFLAGS)"
POCL_CC_FLAGS += POCL_VORTEX_LDFLAGS="$(VX_LDFLAGS)"

HIPCC_FLAGS += -std=c++17
HIPCC_FLAGS += $(CONFIGS)

# Override the install prefix baked into hipcc at chipStar build time
# (otherwise hipcc keeps -L'ing the original build-time --hip-path even
# when chipstar was extracted into a different TOOLDIR). HIP_PATH env
# in toolchain_env.sh covers most of the discovery; the -L below makes
# sure the host link finds the relocated libCHIP.so.
HIPCC_FLAGS += --hip-path=$(CHIPSTAR_PATH)
HIPCC_FLAGS += -L$(CHIPSTAR_PATH)/lib -Wl,-rpath,$(CHIPSTAR_PATH)/lib

# libOpenCL.so (POCL with Vortex device driver linked in) needs
# libvortex.so to resolve at link time. Pass -rpath-link to the linker
# so it can chase the transitive dep without us linking vortex directly.
HIPCC_FLAGS += -Wl,-rpath-link,$(VORTEX_RT_LIB)
HIPCC_FLAGS += -Wl,-rpath-link,$(POCL_PATH)/lib
HIPCC_FLAGS += -Wl,-rpath-link,$(LLVM_VORTEX)/lib

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
	LD_LIBRARY_PATH=$(LLVM_VORTEX)/lib:$(LD_LIBRARY_PATH) $(HIPCC) $(HIPCC_FLAGS) -I. $< -o $@

run-simx: $(PROJECT)
	$(RUNTIME_ARGS) $(MAKE) -C $(VORTEX_RT_SRC)/simx DESTDIR=$(VORTEX_RT_LIB)
	LD_LIBRARY_PATH=$(CHIPSTAR_PATH)/lib:$(POCL_PATH)/lib:$(VORTEX_RT_LIB):$(LLVM_VORTEX)/lib:$(LD_LIBRARY_PATH) $(POCL_CC_FLAGS) VORTEX_DRIVER=simx ./$(PROJECT) $(OPTS)

run-rtlsim: $(PROJECT)
	$(RUNTIME_ARGS) $(MAKE) -C $(VORTEX_RT_SRC)/rtlsim DESTDIR=$(VORTEX_RT_LIB)
	LD_LIBRARY_PATH=$(CHIPSTAR_PATH)/lib:$(POCL_PATH)/lib:$(VORTEX_RT_LIB):$(LLVM_VORTEX)/lib:$(LD_LIBRARY_PATH) $(POCL_CC_FLAGS) VORTEX_DRIVER=rtlsim ./$(PROJECT) $(OPTS)

clean:
	rm -f $(PROJECT) *.o *.vxbin *.dump *.ll *.log *.spv common.h

.PHONY: all run-simx run-rtlsim clean
