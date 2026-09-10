ROOT_DIR := $(realpath ../../..)
include $(ROOT_DIR)/config.mk

PROJECT := dxa_ooo

SIMX_DIR := $(ROOT_DIR)/sim/simx

CONFIGS := $(if $(findstring -DVX_CFG_EXT_DXA_ENABLE,$(CONFIGS)),$(CONFIGS),$(CONFIGS) -DVX_CFG_EXT_DXA_ENABLE)

.PHONY: all run clean

all:
	$(MAKE) -C $(SIMX_DIR) $(SIMX_DIR)/dxa_ooo_test CONFIGS="$(CONFIGS)"

run:
	$(MAKE) -C $(SIMX_DIR) test-dxa-ooo CONFIGS="$(CONFIGS)"

clean:
	rm -f $(SIMX_DIR)/dxa_ooo_test $(SIMX_DIR)/obj/tests/dxa_ooo_test.o $(SIMX_DIR)/obj/tests/dxa_ooo_test.d
