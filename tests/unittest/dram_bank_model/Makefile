ROOT_DIR := $(realpath ../../..)
include $(ROOT_DIR)/config.mk

PROJECT := dram_bank_model

SRC_DIR := $(VORTEX_HOME)/tests/unittest/$(PROJECT)
RAMULATOR_DIR := $(THIRD_PARTY_DIR)/ramulator

SRCS := $(SRC_DIR)/main.cpp $(SIM_COMMON_DIR)/dram_sim.cpp

CXXFLAGS += -I$(SIM_COMMON_DIR)
CXXFLAGS += -isystem $(RAMULATOR_DIR)/ext/spdlog/include -isystem $(RAMULATOR_DIR)/ext/yaml-cpp/include -isystem $(RAMULATOR_DIR)/src
LDFLAGS += -Wl,-rpath,$(RAMULATOR_DIR) -L$(RAMULATOR_DIR) -lramulator
LDFLAGS += -Wl,-rpath,$(THIRD_PARTY_DIR)/ramulator_ext -L$(THIRD_PARTY_DIR)/ramulator_ext -Wl,--push-state,--no-as-needed -lramulator-vortex -Wl,--pop-state

include ../common.mk
