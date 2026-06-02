include ../common.mk

# HOST_ARCH selects the cross-compiler for non-native host builds.
# Cross-arch outputs land in $(DESTDIR)/$(HOST_ARCH)/ alongside the
# native build in $(DESTDIR)/, so both share the same dlopen target name.
HOST_ARCH ?= x86_64

DESTDIR ?= $(CURDIR)/..

SRC_DIR := $(VORTEX_HOME)/sw/runtime/stub

CXXFLAGS += -std=c++17 -Wall -Wextra -Wfatal-errors -Werror
CXXFLAGS += -I$(INC_DIR) -I$(VORTEX_HOME)/sw/kernel/include -I$(ROOT_DIR)/sw -I$(ROOT_DIR)/hw -I$(SW_COMMON_DIR) -I$(RT_COMMON_DIR)
CXXFLAGS += -fPIC
CXXFLAGS += $(CONFIGS)

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

# libvortex.so exports the runtime API and VMManager symbols shared by
# all backend drivers. Unresolved driver symbols are resolved at runtime
# when a backend is dlopen'd.
SRCS := \
	$(SRC_DIR)/vortex.cpp \
	$(VORTEX_HOME)/sw/runtime/graphics.cpp \
	$(RT_COMMON_DIR)/result.cpp \
	$(RT_COMMON_DIR)/device.cpp \
	$(RT_COMMON_DIR)/buffer.cpp \
	$(RT_COMMON_DIR)/queue.cpp \
	$(RT_COMMON_DIR)/event.cpp \
	$(RT_COMMON_DIR)/module.cpp \
	$(RT_COMMON_DIR)/runtime_helpers.cpp \
	$(RT_COMMON_DIR)/legacy_runtime.cpp \
	$(RT_COMMON_DIR)/legacy_utils.cpp \
	$(RT_COMMON_DIR)/legacy_perf.cpp \
	$(RT_COMMON_DIR)/utils.cpp \
	$(RT_COMMON_DIR)/vm.cpp

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
