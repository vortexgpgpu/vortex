include ../common.mk

DESTDIR ?= $(CURDIR)/..

SRC_DIR := $(VORTEX_HOME)/sw/runtime/stub

CXXFLAGS += -std=c++17 -Wall -Wextra -pedantic -Wfatal-errors -Werror
CXXFLAGS += -I$(INC_DIR) -I$(ROOT_DIR)/sw -I$(ROOT_DIR)/hw -I$(SW_COMMON_DIR) -I$(RT_COMMON_DIR)
CXXFLAGS += -fPIC
CXXFLAGS += $(CONFIGS)

LDFLAGS += -shared -pthread -ldl -Wl,-soname,libvortex.so

# vm.cpp lives here so VMManager symbols are exported by libvortex.so
# and consumed by both simx and rtlsim runtime shims via -lvortex
# (no per-driver duplication). vm.cpp depends on RAM (sim/common/mem.h)
# but those symbols stay unresolved in libvortex.so until a sim driver
# is dlopen'd at runtime — same pattern as dcr_read/mem_alloc.
SRCS := $(SRC_DIR)/vortex.cpp $(SRC_DIR)/utils.cpp $(SRC_DIR)/perf.cpp $(RT_COMMON_DIR)/utils.cpp $(RT_COMMON_DIR)/vm.cpp

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