ROOT_DIR := $(realpath ../../..)

TARGET ?= opaesim

XRT_SYN_DIR ?= $(VORTEX_HOME)/hw/syn/xilinx/xrt
XRT_DEVICE_INDEX ?= 0

VORTEX_RT_SRC ?= $(ROOT_DIR)/sw/runtime
VORTEX_RT_LIB ?= $(VORTEX_RT_SRC)
VORTEX_KN_PATH ?= $(ROOT_DIR)/sw/kernel

KERNEL_LIB ?= vortex

# Resolve the toml + CONFIGS overrides into the canonical -D... list, the
# same way sim/simx/Makefile does. Then sniff for extension enables.
XCONFIGS := $(shell python3 $(ROOT_DIR)/ci/gen_config.py --config=$(VORTEX_HOME)/VX_config.toml --cflags='$(CONFIGS) -DXLEN_$(XLEN)')

ifneq (,$(filter -DEXT_C_ENABLE, $(XCONFIGS)))
	C_EXT := c
else
	C_EXT :=
endif

ifeq ($(XLEN),64)
	ifneq (,$(filter -DEXT_V_ENABLE, $(XCONFIGS)))
		VX_CFLAGS += -march=rv64imafd$(C_EXT)v_zve64d -mabi=lp64d # vector extension
	else
		VX_CFLAGS += -march=rv64imafd$(C_EXT) -mabi=lp64d
	endif
	STARTUP_ADDR ?= 0x180000000
else
	ifneq (,$(filter -DEXT_V_ENABLE, $(XCONFIGS)))
		VX_CFLAGS += -march=rv32imaf$(C_EXT)v_zve32f -mabi=ilp32f # vector extension
	else
		VX_CFLAGS += -march=rv32imaf$(C_EXT) -mabi=ilp32f
	endif
	STARTUP_ADDR ?= 0x80000000
endif

LLVM_CFLAGS += --target=riscv$(XLEN)-unknown-elf
LLVM_CFLAGS += --sysroot=$(RISCV_SYSROOT)
LLVM_CFLAGS += --gcc-toolchain=$(RISCV_TOOLCHAIN_PATH)
LLVM_CFLAGS += -Xclang -target-feature -Xclang +xvortex
LLVM_CFLAGS += -Xclang -target-feature -Xclang +zicond
LLVM_CFLAGS += -mllvm -disable-loop-idiom-all # disable memset/memcpy loop idiom
LLVM_CFLAGS += -Wno-unused-command-line-argument
#LLVM_CFLAGS += -mllvm -vortex-branch-divergence=0
#LLVM_CFLAGS += -mllvm -debug -mllvm -print-after-all
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

VX_CFLAGS += -Wall -Wextra -Wfatal-errors -Werror -Wno-unused-command-line-argument
VX_CFLAGS += -O3 -mcmodel=medany -fno-rtti -fno-exceptions -nostartfiles -nostdlib -fdata-sections -ffunction-sections
VX_CFLAGS += -I$(VORTEX_HOME)/sw/kernel/include -I$(ROOT_DIR)/sw -I$(ROOT_DIR)/hw -I$(SW_COMMON_DIR)
VX_CFLAGS += -DXLEN_$(XLEN) -DNDEBUG -D__VORTEX__
VX_CFLAGS += $(CONFIGS)

VX_LIBS += -L$(LIBC_VORTEX)/lib -lm -lc

VX_LIBS += $(LIBCRT_VORTEX)/lib/baremetal/libclang_rt.builtins-riscv$(XLEN).a
#VX_LIBS += -lgcc

VX_LDFLAGS += -Wl,-Bstatic,--gc-sections,-T,$(VORTEX_HOME)/sw/kernel/scripts/link$(XLEN).ld,--defsym=STARTUP_ADDR=$(STARTUP_ADDR) $(VORTEX_KN_PATH)/lib$(KERNEL_LIB).a $(VX_LIBS)

VX_STARTUP_SRC := $(VORTEX_HOME)/sw/kernel/src/vx_start.S
VX_KMU_FLAG := $(if $(filter vortex2,$(KERNEL_LIB)),-DKMU_ENABLE)
VX_APP_OBJS = $(addsuffix .o, $(basename $(notdir $(VX_SRCS))))
KERNEL_STARTUP := $(VORTEX_HOME)/sw/kernel/scripts/kernel_startup.sh

CXXFLAGS += -std=c++17 -Wall -Wextra -pedantic -Wfatal-errors -Werror
CXXFLAGS += -I$(VORTEX_HOME)/sw/runtime/include -I$(ROOT_DIR)/sw -I$(ROOT_DIR)/hw -I$(SW_COMMON_DIR)
CXXFLAGS += $(CONFIGS)

# HOST_ARCH selects the simulated-host compiler for the test binary
# (the .vxbin always builds with the RISC-V toolchain regardless).
# When non-native, the binary is suffixed (e.g. vecadd-aarch64) and
# we link against the cross-compiled stub in $(VORTEX_RT_LIB)/$(HOST_ARCH)/.
# Aligns with sw/runtime/{stub,gem5}/Makefile's HOST_ARCH knob; the
# gem5 ARM e2e test path uses this to produce aarch64 binaries that
# the simulated ARM CPU inside gem5 can execute.
#
# Cross-compiled ELFs embed `/lib/ld-linux-$arch.so.1` as the dynamic
# linker (PT_INTERP). gem5 doesn't have that path on the host, but
# it has a setInterpDir() API that prepends a sysroot to the
# interpreter lookup — the gem5 Python config calls that when
# DRIVER=gem5-aarch64. Keep the default INTERP here so that mechanism
# can do the redirection cleanly. (Earlier versions used
# `-Wl,--dynamic-linker=` to rewrite PT_INTERP, but that interacts
# badly with setInterpDir's prepend logic.)
HOST_ARCH ?= x86_64
ifeq ($(HOST_ARCH),x86_64)
    PROJECT_SUFFIX :=
    RT_LIB_DIR := $(VORTEX_RT_LIB)
else ifeq ($(HOST_ARCH),aarch64)
    CXX := aarch64-linux-gnu-g++
    PROJECT_SUFFIX := -aarch64
    RT_LIB_DIR := $(VORTEX_RT_LIB)/aarch64
else ifeq ($(HOST_ARCH),armhf)
    CXX := arm-linux-gnueabihf-g++
    PROJECT_SUFFIX := -armhf
    RT_LIB_DIR := $(VORTEX_RT_LIB)/armhf
else
    $(error HOST_ARCH must be one of: x86_64, aarch64, armhf (got $(HOST_ARCH)))
endif

LDFLAGS += -L$(RT_LIB_DIR) -lvortex

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

CONFIG_STAMP = config.stamp

# HOST_ARCH-suffixed binary name (vecadd, vecadd-aarch64, …) so
# x86 and cross-compiled variants coexist in the same dir.
APP := $(PROJECT)$(PROJECT_SUFFIX)

all: $(APP) kernel.vxbin kernel.dump

# Force rebuild when CONFIGS (defines) change between runs.
# PID-suffixed tmp + tolerate concurrent invocations across test
# Makefiles + blackbox.sh-driven runtime builds (mv -f).
$(CONFIG_STAMP): FORCE
	@TMP=$@.tmp.$$$$ ; \
	 printf '%s\n' '$(VX_CFLAGS)' '$(CXXFLAGS)' '$(LDFLAGS)' > $$TMP ; \
	 if ! cmp -s $$TMP $@ 2>/dev/null; then \
	   mv -f $$TMP $@ ; \
	 else \
	   rm -f $$TMP ; \
	 fi
FORCE:

kernel.dump: kernel.elf
	$(VX_DP) -D $< > $@

kernel.vxbin: kernel.elf
	OBJCOPY=$(VX_CP) $(VORTEX_HOME)/sw/kernel/scripts/vxbin.py $< $@

$(VORTEX_KN_PATH)/lib$(KERNEL_LIB).a:
	$(MAKE) -C $(VORTEX_KN_PATH)

RUNTIME_ARGS = CONFIGS="$(CONFIGS)" $(if $(DEBUG),DEBUG=$(DEBUG)) $(if $(PERF),PERF=$(PERF)) $(if $(SCOPE),SCOPE=$(SCOPE))

$(VORTEX_RT_LIB)/libvortex.so:
	$(RUNTIME_ARGS) $(MAKE) -C $(VORTEX_RT_SRC)/stub DESTDIR=$(VORTEX_RT_LIB)

ifneq ($(filter %.S,$(VX_SRCS)),)
kernel.elf: $(VX_SRCS) $(VORTEX_KN_PATH)/lib$(KERNEL_LIB).a $(CONFIG_STAMP)
	$(VX_CXX) $(VX_CFLAGS) $(filter-out $(CONFIG_STAMP),$^) $(VX_LDFLAGS) -o $@
else
vx_start.o: $(VX_SRCS) $(VORTEX_KN_PATH)/lib$(KERNEL_LIB).a $(CONFIG_STAMP)
	$(VX_CXX) $(VX_CFLAGS) -c $(VX_SRCS)
	$(VX_CXX) $(VX_CFLAGS) -DNEED_GP -DNEED_TLS -DNEED_INITFINI $(VX_KMU_FLAG) -c $(VX_STARTUP_SRC) -o $@
	$(VX_CXX) $(VX_CFLAGS) $@ $(VX_APP_OBJS) $(VX_LDFLAGS) -o $@.elf
	$(VX_CXX) $(VX_CFLAGS) $$($(KERNEL_STARTUP) $(VX_DP) $@.elf) $(VX_KMU_FLAG) -c $(VX_STARTUP_SRC) -o $@ && rm -f $@.elf

kernel.elf: vx_start.o $(VX_SRCS) $(VORTEX_KN_PATH)/lib$(KERNEL_LIB).a $(CONFIG_STAMP)
	$(VX_CXX) $(VX_CFLAGS) vx_start.o $(VX_APP_OBJS) $(VX_LDFLAGS) -o $@
endif

$(APP): $(SRCS) $(RT_LIB_DIR)/libvortex.so $(CONFIG_STAMP)
	$(CXX) $(CXXFLAGS) $(filter-out $(CONFIG_STAMP),$^) $(LDFLAGS) -o $@

# Cross-compiled stub for non-native HOST_ARCH. Native (x86_64)
# is built by $(VORTEX_RT_LIB)/libvortex.so rule below.
ifneq ($(HOST_ARCH),x86_64)
$(RT_LIB_DIR)/libvortex.so:
	$(RUNTIME_ARGS) $(MAKE) -C $(VORTEX_RT_SRC)/stub HOST_ARCH=$(HOST_ARCH) DESTDIR=$(VORTEX_RT_LIB)
endif

run-simx: $(PROJECT) kernel.vxbin
	$(RUNTIME_ARGS) $(MAKE) -C $(VORTEX_RT_SRC)/simx DESTDIR=$(VORTEX_RT_LIB)
	LD_LIBRARY_PATH=$(VORTEX_RT_LIB):$(LD_LIBRARY_PATH) VORTEX_DRIVER=simx ./$(PROJECT) $(OPTS)

run-rtlsim: $(PROJECT) kernel.vxbin
	$(RUNTIME_ARGS) $(MAKE) -C $(VORTEX_RT_SRC)/rtlsim DESTDIR=$(VORTEX_RT_LIB)
	LD_LIBRARY_PATH=$(VORTEX_RT_LIB):$(LD_LIBRARY_PATH) VORTEX_DRIVER=rtlsim ./$(PROJECT) $(OPTS)

run-opae: $(PROJECT) kernel.vxbin
	$(RUNTIME_ARGS) $(MAKE) -C $(VORTEX_RT_SRC)/opae DESTDIR=$(VORTEX_RT_LIB)
	SCOPE_JSON_PATH=$(VORTEX_RT_LIB)/scope.json OPAE_DRV_PATHS=$(OPAE_DRV_PATHS) LD_LIBRARY_PATH=$(VORTEX_RT_LIB):$(LD_LIBRARY_PATH) VORTEX_DRIVER=opae ./$(PROJECT) $(OPTS)

run-xrt: $(PROJECT) kernel.vxbin
	$(RUNTIME_ARGS) $(MAKE) -C $(VORTEX_RT_SRC)/xrt DESTDIR=$(VORTEX_RT_LIB)
ifeq ($(TARGET), hw)
	SCOPE_JSON_PATH=$(FPGA_BIN_DIR)/scope.json XRT_INI_PATH=$(VORTEX_RT_SRC)/xrt/xrt.ini EMCONFIG_PATH=$(FPGA_BIN_DIR) XRT_DEVICE_INDEX=$(XRT_DEVICE_INDEX) XRT_XCLBIN_PATH=$(FPGA_BIN_DIR)/vortex_afu.xclbin LD_LIBRARY_PATH=$(XILINX_XRT)/lib:$(VORTEX_RT_LIB):$(LD_LIBRARY_PATH) VORTEX_DRIVER=xrt ./$(PROJECT) $(OPTS)
else ifeq ($(TARGET), hw_emu)
	SCOPE_JSON_PATH=$(FPGA_BIN_DIR)/scope.json XCL_EMULATION_MODE=$(TARGET) XRT_INI_PATH=$(VORTEX_RT_SRC)/xrt/xrt.ini EMCONFIG_PATH=$(FPGA_BIN_DIR) XRT_DEVICE_INDEX=$(XRT_DEVICE_INDEX) XRT_XCLBIN_PATH=$(FPGA_BIN_DIR)/vortex_afu.xclbin LD_LIBRARY_PATH=$(XILINX_XRT)/lib:$(VORTEX_RT_LIB):$(LD_LIBRARY_PATH) VORTEX_DRIVER=xrt ./$(PROJECT) $(OPTS)
else
	SCOPE_JSON_PATH=$(VORTEX_RT_LIB)/scope.json LD_LIBRARY_PATH=$(XILINX_XRT)/lib:$(VORTEX_RT_LIB):$(LD_LIBRARY_PATH) VORTEX_DRIVER=xrt ./$(PROJECT) $(OPTS)
endif

.depend: $(SRCS)
	$(CXX) $(CXXFLAGS) -MM $^ > .depend;

clean-kernel:
	rm -rf *.elf *.vxbin *.dump

clean-host:
	rm -rf $(PROJECT) *.o *.log .depend

clean: clean-kernel clean-host
	rm -f $(CONFIG_STAMP) $(CONFIG_STAMP).tmp

ifneq ($(MAKECMDGOALS),clean)
    -include .depend
endif
