# Shared build rules for tests/hip/*.
#
# The HIP toolchain referenced below is the one defined in
# docs/proposals/hip_support_proposal.md. Pieces land progressively as
# the proposal is implemented; until each piece exists the corresponding
# rule will fail with a "command not found" error against the missing
# tool. That is by design — these test directories serve both as
# reference HIP source and as the integration target for the toolchain
# work itself.

# Tooling — provided once the HIPVortex driver and runtime are built
# from the hip_vortex s/w stack and installed into $(TOOLDIR). The Vortex
# tree (sw/) is not touched: the HIP runtime, headers, and wrapper
# script all live in ~/dev/hip_vortex.
HIP_INSTALL_PATH ?= $(TOOLDIR)/hip-vortex
HIPCC_VORTEX     ?= $(HIP_INSTALL_PATH)/bin/hipcc-vortex
HIP_INCLUDE_PATH ?= $(HIP_INSTALL_PATH)/include
HIP_LIB_PATH     ?= $(HIP_INSTALL_PATH)/lib

HIPCC_FLAGS += --offload-arch=vortex
HIPCC_FLAGS += -std=c++17 -O2
HIPCC_FLAGS += -I$(HIP_INCLUDE_PATH)
HIPCC_FLAGS += $(CONFIGS)

OPTS ?= -n64

VORTEX_RT_SRC ?= $(ROOT_DIR)/sw/runtime
VORTEX_RT_LIB ?= $(VORTEX_RT_SRC)

RUNTIME_ARGS = CONFIGS="$(CONFIGS)" $(if $(DEBUG),DEBUG=$(DEBUG)) $(if $(PERF),PERF=$(PERF))

all: $(PROJECT)

# Single-source HIP compile. The HIPVortex toolchain in llvm_vortex splits
# the input into host + Vortex device passes, lowers the device pass
# through MLIR (Path B) or directly through LLVM IR (Path A), and emits
# a fat ELF. libhip_vortex.so wraps vortex_runtime at load time.
$(PROJECT): $(SRCS) common.h
	$(HIPCC_VORTEX) $(HIPCC_FLAGS) $< -L$(HIP_LIB_PATH) -lhip_vortex -o $@

$(VORTEX_RT_LIB)/libvortex.so:
	$(RUNTIME_ARGS) $(MAKE) -C $(VORTEX_RT_SRC)/stub DESTDIR=$(VORTEX_RT_LIB)

run-simx: $(PROJECT) $(VORTEX_RT_LIB)/libvortex.so
	$(RUNTIME_ARGS) $(MAKE) -C $(VORTEX_RT_SRC)/simx DESTDIR=$(VORTEX_RT_LIB)
	LD_LIBRARY_PATH=$(VORTEX_RT_LIB):$(HIP_LIB_PATH):$(LD_LIBRARY_PATH) \
	    VORTEX_DRIVER=simx ./$(PROJECT) $(OPTS)

run-rtlsim: $(PROJECT) $(VORTEX_RT_LIB)/libvortex.so
	$(RUNTIME_ARGS) $(MAKE) -C $(VORTEX_RT_SRC)/rtlsim DESTDIR=$(VORTEX_RT_LIB)
	LD_LIBRARY_PATH=$(VORTEX_RT_LIB):$(HIP_LIB_PATH):$(LD_LIBRARY_PATH) \
	    VORTEX_DRIVER=rtlsim ./$(PROJECT) $(OPTS)

clean:
	rm -f $(PROJECT) *.o *.vxbin *.dump *.ll *.log common.h

.PHONY: all run-simx run-rtlsim clean
