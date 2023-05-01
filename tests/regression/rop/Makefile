PROJECT = rop

OPTS ?=

VX_CFLAGS += -I$(VORTEX_KN_PATH)/../sim/common -I$(VORTEX_KN_PATH)/../third_party

CXXFLAGS += -I$(VORTEX_KN_PATH)/../sim/common -I$(VORTEX_KN_PATH)/../third_party

LDFLAGS += $(VORTEX_KN_PATH)/../third_party/cocogfx/libcocogfx.a -lpng -lz

SRCS = main.cpp

VX_SRCS = kernel.cpp

include ../common.mk