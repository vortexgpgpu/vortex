# Which card a FireSim run is for, decided the same way wherever it is asked --
# the runtime build, run-firesim, blackbox.sh's driver build -- from what the
# caller already has: TARGET and, for an image target, FPGA_BIN_DIR.
#
#   TARGET=hw|hw_emu|sim   the card hw/syn/firesim recorded beside the image
#                          ($(FPGA_BIN_DIR)/firesim.card)
#   TARGET=xrtsim|avedsim  no image; the U55C, or the V80 (FIRESIM_CARD overrides)
#   TARGET empty           xrtsim, or avedsim when FIRESIM_CARD names an aved card
#
# Sets FIRESIM_CARD and FIRESIM_PLATFORM (vitis or aved), and TARGET when empty.

FIRESIM_GEN := python3 $(VORTEX_HOME)/hw/scripts/gen_platform.py
firesim_platform_of = $(shell $(FIRESIM_GEN) --card=$(1) --emit=firesim-mk 2>/dev/null | sed -n 's/^FIRESIM_PLATFORM := //p')

ifeq ($(strip $(TARGET)),)
FIRESIM_CARD ?= u55c
override TARGET := $(if $(filter aved,$(call firesim_platform_of,$(FIRESIM_CARD))),avedsim,xrtsim)
else ifneq ($(filter $(TARGET),xrtsim avedsim),)
FIRESIM_CARD ?= $(if $(filter avedsim,$(TARGET)),v80,u55c)
else ifeq ($(filter $(TARGET),hw hw_emu sim),)
$(error TARGET=$(TARGET) is not a FireSim target: use xrtsim, hw_emu or hw on an XRT card, \
        avedsim, sim or hw on an aved card)
else
FIRESIM_CARD ?= $(shell cat $(FPGA_BIN_DIR)/firesim.card 2>/dev/null)
ifeq ($(FIRESIM_CARD),)
$(error TARGET=$(TARGET) runs an image: set FPGA_BIN_DIR to the bin directory hw/syn/firesim built \
        (it holds firesim.card), or use TARGET=xrtsim|avedsim)
endif
endif

FIRESIM_PLATFORM := $(call firesim_platform_of,$(FIRESIM_CARD))
ifeq ($(FIRESIM_PLATFORM),)
$(error cannot resolve the FireSim platform of card '$(FIRESIM_CARD)')
endif

FIRESIM_TARGETS := $(if $(filter aved,$(FIRESIM_PLATFORM)),avedsim sim hw,xrtsim hw_emu hw)
ifeq ($(filter $(TARGET),$(FIRESIM_TARGETS)),)
$(error TARGET=$(TARGET) is not a FireSim target for card $(FIRESIM_CARD): use one of $(FIRESIM_TARGETS))
endif
