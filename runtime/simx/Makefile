include ../common.mk

DESTDIR ?= $(CURDIR)

SRC_DIR := $(VORTEX_HOME)/runtime/simx

CXXFLAGS += -std=c++11 -Wall -Wextra -Wfatal-errors
CXXFLAGS += -fPIC -Wno-maybe-uninitialized
CXXFLAGS += -I$(INC_DIR) -I../common -I$(ROOT_DIR)/hw -I$(SIM_DIR)/simx -I$(COMMON_DIR) -I$(SIM_DIR)/common
CXXFLAGS += $(CONFIGS)
CXXFLAGS += -DXLEN_$(XLEN)

LDFLAGS += -shared -pthread
LDFLAGS += -L$(DESTDIR) -lsimx

SRCS := $(SRC_DIR)/vortex.cpp $(COMMON_DIR)/utils.cpp

# Debugigng
ifdef DEBUG
	CXXFLAGS += -g -O0
else
	CXXFLAGS += -O2 -DNDEBUG
endif

PROJECT := libvortex.so

all: $(DESTDIR)/$(PROJECT)

$(DESTDIR)/$(PROJECT): $(SRCS)
	DESTDIR=$(DESTDIR) $(MAKE) -C $(ROOT_DIR)/sim/simx $(DESTDIR)/libsimx.so
	$(CXX) $(CXXFLAGS) $^ $(LDFLAGS) -o $@

clean:
	DESTDIR=$(DESTDIR) $(MAKE) -C $(ROOT_DIR)/sim/simx clean
	rm -rf $(DESTDIR)/$(PROJECT) *.o