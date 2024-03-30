include ../common.mk

DESTDIR ?= $(CURDIR)

SRC_DIR := $(VORTEX_HOME)/runtime/stub

CXXFLAGS += -std=c++11 -O2 -Wall -Wextra -pedantic -Wfatal-errors
CXXFLAGS += -I$(INC_DIR) -I$(COMMON_DIR) -I$(ROOT_DIR)/hw -I$(SIM_DIR)/common
CXXFLAGS += -fPIC

LDFLAGS += -shared -pthread

SRCS := $(SRC_DIR)/vortex.cpp $(COMMON_DIR)/utils.cpp

PROJECT := libvortex.so

all: $(PROJECT)

$(PROJECT): $(SRCS) 
	$(CXX) $(CXXFLAGS) $^ $(LDFLAGS) -o $@

clean:
	rm -rf $(PROJECT) obj_dir