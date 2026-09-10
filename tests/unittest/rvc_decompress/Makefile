ROOT_DIR := $(realpath ../../..)
include $(ROOT_DIR)/config.mk

PROJECT := rvc_decompress

SIMX_DIR := $(ROOT_DIR)/sim/simx

.PHONY: all run clean

all:
	$(MAKE) -C $(SIMX_DIR) $(SIMX_DIR)/rvc_decompress_test CONFIGS="$(CONFIGS)"

run:
	$(MAKE) -C $(SIMX_DIR) test-rvc-decompress CONFIGS="$(CONFIGS)"

clean:
	rm -f $(SIMX_DIR)/rvc_decompress_test $(SIMX_DIR)/obj/tests/rvc_decompress_test.o $(SIMX_DIR)/obj/tests/rvc_decompress_test.d
