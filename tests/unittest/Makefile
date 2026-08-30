ROOT_DIR := $(realpath ../..)
include $(ROOT_DIR)/config.mk

PROJECTS := vx_malloc dxa_ooo gfx_binsort gfx_tex_sw gfx_rast_sw gfx_msaa gfx_om_mrt

all:
	$(foreach p,$(PROJECTS),$(MAKE) -C $(p) &&) true

run:
	$(foreach p,$(PROJECTS),$(MAKE) -C $(p) run &&) true

clean:
	$(foreach p,$(PROJECTS),$(MAKE) -C $(p) clean &&) true
