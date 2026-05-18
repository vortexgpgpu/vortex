ROOT_DIR := $(realpath ../../..)
include $(ROOT_DIR)/config.mk

PROJECT := cp_sim

SRC_DIR := $(VORTEX_HOME)/hw/unittest/$(PROJECT)
SIM_COMMON := $(VORTEX_HOME)/sim/common

# Pure C++ unit test — no Verilator. The CommandProcessor C++ class
# under test (sim/common/CommandProcessor.cpp) has no RTL dependencies
# beyond its hooks; we provide mock hooks from main.cpp.

CXXFLAGS := -std=c++17 -Wall -Wextra -Wpedantic -Wfatal-errors -Werror
CXXFLAGS += -I$(SIM_COMMON) -I$(SRC_DIR)

SRCS := $(SRC_DIR)/main.cpp $(SIM_COMMON)/CommandProcessor.cpp

DESTDIR ?= $(CURDIR)
PROJECT_BIN := $(DESTDIR)/$(PROJECT).bin

ifdef DEBUG
	CXXFLAGS += -g -O0
else
	CXXFLAGS += -O2 -DNDEBUG
endif

all: $(PROJECT_BIN)

$(PROJECT_BIN): $(SRCS)
	$(CXX) $(CXXFLAGS) $^ -o $@

run: $(PROJECT_BIN)
	$<

clean:
	rm -f $(PROJECT_BIN)

.PHONY: all run clean
