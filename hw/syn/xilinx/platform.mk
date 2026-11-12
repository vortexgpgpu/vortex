# Card facts for the Xilinx flows, resolved from hw/syn/xilinx/platforms/<card>.toml.
#
# The including flow sets PLATFORM_GEN_ARGS (how it names the card) and
# BUILD_DIR before including this. $(shell) discards exit codes, so the
# generator's status is captured explicitly: a descriptor error must stop the
# build here, not surface later as a design built for the wrong card.

PLATFORM_MK := $(BUILD_DIR)/platform.mk
PLATFORM_GEN := python3 $(VORTEX_HOME)/hw/scripts/gen_platform.py $(PLATFORM_GEN_ARGS)

PLATFORM_GEN_RC := $(shell mkdir -p $(BUILD_DIR) && $(PLATFORM_GEN) --emit=mk -o $(PLATFORM_MK) 2> $(PLATFORM_MK).err; echo $$?)
ifneq ($(PLATFORM_GEN_RC),0)
$(error $(shell cat $(PLATFORM_MK).err))
endif
include $(PLATFORM_MK)

ifneq ($(filter $(HOST_TAG),$(PLATFORM_HOST_REFUSE)),)
$(error HOST_TAG=$(HOST_TAG) is refused on card $(PLATFORM_CARD): $(PLATFORM_HOST_REFUSE_WHY))
endif
