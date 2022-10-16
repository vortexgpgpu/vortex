PROJECT = rop

OPTS ?=

VX_CFLAGS += -DENABLE_SW -I$(VORTEX_KN_PATH)/../sim/common -I$(VORTEX_KN_PATH)/../third_party

CXXFLAGS += -I$(VORTEX_KN_PATH)/../sim/common -I$(VORTEX_KN_PATH)/../third_party

LDFLAGS += -L/homes/tinebp/tools/boost/lib $(VORTEX_KN_PATH)/../third_party/cocogfx/libcocogfx.a -lpng -lz -lboost_serialization

SRCS = main.cpp utils.cpp

VX_SRCS = kernel.cpp

include ../common.mk