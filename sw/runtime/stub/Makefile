include ../common.mk

# HOST_ARCH switch — when building for a non-native simulated host
# (e.g. running x86 gem5 with an aarch64 simulated CPU), select the
# matching cross-compiler. Aligns with sw/runtime/gem5/Makefile's
# HOST_ARCH knob; cross-arch builds land in $(DESTDIR)/$(HOST_ARCH)/
# so the same dlopen target name (libvortex.so) can coexist with the
# native build in $(DESTDIR)/.
HOST_ARCH ?= x86_64

DESTDIR ?= $(CURDIR)/..

SRC_DIR := $(VORTEX_HOME)/sw/runtime/stub

CXXFLAGS += -std=c++17 -Wall -Wextra -Wfatal-errors -Werror
CXXFLAGS += -I$(INC_DIR) -I$(ROOT_DIR)/sw -I$(ROOT_DIR)/hw -I$(SW_COMMON_DIR) -I$(RT_COMMON_DIR)
CXXFLAGS += -fPIC

LDFLAGS += -shared -pthread -ldl -Wl,-soname,libvortex.so
# Look for libvortex-<NAME>.so siblings in the same directory libvortex.so
# itself lives in (so the dlopen at vx_device_open time finds them).
LDFLAGS += -Wl,-rpath,'$$ORIGIN'

ifeq ($(HOST_ARCH),x86_64)
    CXX := g++
    OUT_DIR := $(DESTDIR)
else ifeq ($(HOST_ARCH),aarch64)
    CXX := aarch64-linux-gnu-g++
    OUT_DIR := $(DESTDIR)/aarch64
else ifeq ($(HOST_ARCH),armhf)
    CXX := arm-linux-gnueabihf-g++
    OUT_DIR := $(DESTDIR)/armhf
else
    $(error HOST_ARCH must be one of: x86_64, aarch64, armhf (got $(HOST_ARCH)))
endif

# Dispatcher library = vortex2.h runtime (C++ classes) +
#                      legacy_runtime.cpp wrappers (vortex.h -> vortex2.h) +
#                      legacy utility helpers +
#                      thin stub/vortex.cpp glue (currently just for the
#                      build target — the real entry points live in
#                      common/).
SRCS := \
	$(SRC_DIR)/vortex.cpp \
	$(RT_COMMON_DIR)/vx_result.cpp \
	$(RT_COMMON_DIR)/vx_device.cpp \
	$(RT_COMMON_DIR)/vx_buffer.cpp \
	$(RT_COMMON_DIR)/vx_queue.cpp \
	$(RT_COMMON_DIR)/vx_event.cpp \
	$(RT_COMMON_DIR)/vx_runtime_helpers.cpp \
	$(RT_COMMON_DIR)/legacy_runtime.cpp \
	$(RT_COMMON_DIR)/legacy_utils.cpp \
	$(RT_COMMON_DIR)/legacy_perf.cpp \
	$(RT_COMMON_DIR)/utils.cpp

# Debugging
ifdef DEBUG
	CXXFLAGS += -g -O0
else
	CXXFLAGS += -O2 -DNDEBUG
endif

PROJECT := libvortex.so

all: $(OUT_DIR)/$(PROJECT)

$(OUT_DIR)/$(PROJECT): $(SRCS)
	@mkdir -p $(OUT_DIR)
	$(CXX) $(CXXFLAGS) $^ $(LDFLAGS) -o $@

clean:
	rm -f $(DESTDIR)/$(PROJECT) $(DESTDIR)/aarch64/$(PROJECT) $(DESTDIR)/armhf/$(PROJECT)

.PHONY: all clean
