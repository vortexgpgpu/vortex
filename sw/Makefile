ROOT_DIR := $(realpath ..)
include $(ROOT_DIR)/config.mk

all: kernel runtime

kernel:
	$(MAKE) -C kernel

runtime:
	$(MAKE) -C runtime

clean:
	$(MAKE) -C kernel clean
	$(MAKE) -C runtime clean

.PHONY: all kernel runtime clean
