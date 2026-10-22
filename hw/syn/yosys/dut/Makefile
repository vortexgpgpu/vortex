# Per-DUT entry point for the Yosys/OpenSTA flow.
#
# One Makefile, no subdirectories: a DUT is a top module plus an include path
# plus defines (see catalog.mk), forwarded to ../Makefile. The same command a
# human runs is the one asic_gate runs, so the two cannot diverge.
#
#   make -C hw/syn/yosys/dut gfx                  synthesize + STA (TARGET=timing)
#   make -C hw/syn/yosys/dut tcu TARGET=synthesis synthesis only
#   make -C hw/syn/yosys/dut om CLOCK_FREQ=500    override the target clock
#   make -C hw/syn/yosys/dut om CONFIGS=-DFOO     extra defines, on top of the DUT's
#   make -C hw/syn/yosys/dut core CONFIGS=-DVX_CFG_NUM_THREADS=4   resize the DUT
#   make -C hw/syn/yosys/dut list                 what is available
#
# The DUT's defines go down as EXTRA_CONFIGS, never as CONFIGS: ../Makefile
# accumulates its own mandatory defines (SYNTHESIS/ASIC/YOSYS) with `+=`, and a
# CONFIGS on a make command line overrides that whole chain silently.
#
# PREFIX defaults to the DUT name so each DUT gets its own build tree. That is
# not cosmetic: hw/syn/yosys/Makefile caches $(BUILD_DIR)/src and does not
# regenerate it when EXTRA_INCLUDE changes, so sharing one tree across DUTs
# silently reuses the previous DUT's source set. Override it to keep an
# automated sweep's trees away from a hand-run build of the same DUT (ci/
# asic_gate.py passes PREFIX=asic_gate_<build-id>).

ROOT_DIR := $(realpath ../../../..)
include $(ROOT_DIR)/config.mk

include catalog.mk

TARGET ?= timing
PREFIX ?=

# CONFIGS layers ON TOP OF the DUT's own defines, so a macro the caller sets
# REPLACES the catalog's value rather than sitting beside it. Dropping the
# losing copy is not tidiness: verilator's preprocessor makes a redefinition
# with a different value an error, so `CONFIGS=-DVX_CFG_NUM_THREADS=4` against a
# catalog entry that says 16 used to fail before synthesis instead of resizing
# the DUT.
define_name  = $(firstword $(subst =, ,$1))
last_defines = $(foreach n,$(sort $(foreach d,$1,$(call define_name,$d))),\
                 $(lastword $(filter $n $n=%,$1)))

.PHONY: list $(DUTS)

$(DUTS):
	$(MAKE) -C .. $(TARGET) PREFIX=$(or $(PREFIX),$@) \
	  TOP_LEVEL_ENTITY=$($@_TOP) \
	  EXTRA_INCLUDE="$($@_INC)" \
	  EXTRA_CONFIGS="$(call last_defines,$($@_CFG) $(CONFIGS))"

list:
	@echo $(DUTS) | tr ' ' '\n'
