PROJECT = draw3d

OPTS ?=

VX_CFLAGS += -I$(VORTEX_KN_PATH)/../sim/common -I$(VORTEX_KN_PATH)/../third_party

CXXFLAGS += -I$(VORTEX_KN_PATH)/../sim/common -I$(VORTEX_KN_PATH)/../third_party

LDFLAGS += $(VORTEX_KN_PATH)/../third_party/cocogfx/libcocogfx.a -lpng -lz -lboost_serialization

SRCS = main.cpp $(VORTEX_KN_PATH)/../sim/common/gfxutil.cpp

VX_SRCS = kernel.cpp $(VORTEX_KN_PATH)/../sim/common/graphics.cpp

include ../common.mk

graphics.ll: $(VX_SRCS)
	$(VX_CXX) $(VX_CFLAGS) -mllvm -debug-pass=Arguments $(VX_SRCS) $(VX_LDFLAGS) -S -emit-llvm

graphics.pass: graphics.ll
	$(LLVM_VORTEX)/bin/llc -O3 -march=riscv32 -target-abi=ilp32f -mcpu=generic-rv32 -mattr=+m,+f,+vortex -float-abi=hard -code-model=small -print-after-all -debug-pass=Executions graphics.ll > graphics.pass 2>&1
