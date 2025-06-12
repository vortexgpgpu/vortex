include ../common.mk

DESTDIR ?= $(CURDIR)/..

SRC_DIR := $(VORTEX_HOME)/runtime/stub

CXXFLAGS += -std=c++17 -Wall -Wextra -pedantic -Wfatal-errors
CXXFLAGS += -I$(INC_DIR) -I$(ROOT_DIR)/hw -I$(SW_COMMON_DIR) -I$(RT_COMMON_DIR)
CXXFLAGS += -fPIC

LDFLAGS += -shared -pthread -ldl

SRCS := $(SRC_DIR)/vortex.cpp $(SRC_DIR)/utils.cpp

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