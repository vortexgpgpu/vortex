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

CXXFLAGS += -std=c++17 -Wall -Wextra -pedantic -Wfatal-errors -Werror
CXXFLAGS += -I$(INC_DIR) -I$(ROOT_DIR)/sw -I$(ROOT_DIR)/hw -I$(SW_COMMON_DIR) -I$(RT_COMMON_DIR)
CXXFLAGS += -fPIC

LDFLAGS += -shared -pthread -ldl -Wl,-soname,libvortex.so

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

SRCS := $(SRC_DIR)/vortex.cpp $(SRC_DIR)/utils.cpp $(SRC_DIR)/perf.cpp $(RT_COMMON_DIR)/utils.cpp

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