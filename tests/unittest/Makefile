ROOT_DIR := $(realpath ../..)
include $(ROOT_DIR)/config.mk

PROJECTS := vx_malloc gfx_binsort gfx_tex_sw

all:
	$(foreach p,$(PROJECTS),$(MAKE) -C $(p) &&) true

run:
	$(foreach p,$(PROJECTS),$(MAKE) -C $(p) run &&) true

clean:
	$(foreach p,$(PROJECTS),$(MAKE) -C $(p) clean &&) true
