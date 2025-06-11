include ../common.mk

DESTDIR ?= $(CURDIR)/..

SRC_DIR := $(VORTEX_HOME)/runtime/simx

CXXFLAGS += -std=c++17 -Wall -Wextra -Wfatal-errors
CXXFLAGS += -fPIC -Wno-maybe-uninitialized
CXXFLAGS += -I$(INC_DIR) -I../common -I$(ROOT_DIR)/hw -I$(SIM_DIR)/simx -I$(COMMON_DIR) -I$(COMMON_RT_DIR)
CXXFLAGS += $(CONFIGS)
CXXFLAGS += -DXLEN_$(XLEN)

LDFLAGS += -shared -pthread
LDFLAGS += -L$(DESTDIR) -lsimx

SRCS := $(SRC_DIR)/vortex.cpp

# Debugging
ifdef DEBUG
	CXXFLAGS += -g -O0
else
	CXXFLAGS += -O2 -DNDEBUG
endif

PROJECT := libvortex-simx.so

.PHONY: all clean force clean-driver clean-runtime

all: $(DESTDIR)/$(PROJECT)

driver: $(DESTDIR)/libsimx.so

$(DESTDIR)/libsimx.so: force
	DESTDIR=$(DESTDIR) $(MAKE) -C $(ROOT_DIR)/sim/simx $(DESTDIR)/libsimx.so

$(DESTDIR)/$(PROJECT): $(SRCS) $(DESTDIR)/libsimx.so
	$(CXX) $(CXXFLAGS) $(SRCS) $(LDFLAGS) -o $@

clean-driver:
	DESTDIR=$(DESTDIR) $(MAKE) -C $(ROOT_DIR)/sim/simx clean-lib

clean-runtime:
	rm -f $(DESTDIR)/$(PROJECT)

clean: clean-driver clean-runtime

.PHONY: all driver clean-driver clean-runtime clean
