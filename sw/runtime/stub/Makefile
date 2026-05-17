include ../common.mk

DESTDIR ?= $(CURDIR)/..

SRC_DIR := $(VORTEX_HOME)/sw/runtime/stub

CXXFLAGS += -std=c++17 -Wall -Wextra -Wfatal-errors -Werror
CXXFLAGS += -I$(INC_DIR) -I$(ROOT_DIR)/sw -I$(ROOT_DIR)/hw -I$(SW_COMMON_DIR) -I$(RT_COMMON_DIR)
CXXFLAGS += -fPIC

LDFLAGS += -shared -pthread -ldl -Wl,-soname,libvortex.so
# Look for libvortex-<NAME>.so siblings in the same directory libvortex.so
# itself lives in (so the dlopen at vx_device_open time finds them).
LDFLAGS += -Wl,-rpath,'$$ORIGIN'

# Dispatcher library = vortex2.h runtime (C++ classes) +
#                      vortex_legacy.cpp wrappers (vortex.h -> vortex2.h) +
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

all: $(DESTDIR)/$(PROJECT)

$(DESTDIR)/$(PROJECT): $(SRCS)
	$(CXX) $(CXXFLAGS) $^ $(LDFLAGS) -o $@

clean:
	rm -f $(DESTDIR)/$(PROJECT)

.PHONY: all clean
