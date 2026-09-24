ROOT_DIR := $(realpath ../..)
include $(ROOT_DIR)/config.mk

DESTDIR ?= $(CURDIR)
SRC_DIR := $(VORTEX_HOME)/sim/firesim

PROJECT := firesim

SIMIF_DIR := $(FIRESIM_PATH)/sim/midas/src/main/cc

DESIGN ?= VortexTarget
TARGET_CONFIG ?= VortexConfig

# The card's FireSim platform and platform config come from its descriptor, the
# same ones hw/syn/firesim elaborated the target with.
FIRESIM_CARD ?= u55c
FIRESIM_MK := $(DESTDIR)/firesim_platform.mk
FIRESIM_GEN_RC := $(shell mkdir -p $(DESTDIR) && python3 $(VORTEX_HOME)/hw/scripts/gen_platform.py --card=$(FIRESIM_CARD) --emit=firesim-mk -o $(FIRESIM_MK) 2> $(FIRESIM_MK).err; echo $$?)
ifneq ($(FIRESIM_GEN_RC),0)
$(error $(shell cat $(FIRESIM_MK).err))
endif
include $(FIRESIM_MK)

# Golden Gate writes the generated header here; the driver objects are built
# alongside it so a rebuild of the target invalidates them.
GEN_DIR := $(FIRESIM_PATH)/sim/output/$(PLATFORM)/$(PLATFORM)-midasexamples-$(DESIGN)-$(TARGET_CONFIG)-$(PLATFORM_CONFIG)/build

# The driver is built as a shared library rather than FireSim's usual
# executable, because the Vortex runtime loads it. Upstream's driver.mk only
# emits executables, so its sub-make is invoked directly with our own source
# list. That also keeps the test harnesses out: each defines create_simulation,
# and linking one alongside firesim_sim.cpp would be a duplicate symbol.
DRIVER_SRCS := $(SRC_DIR)/firesim_sim.cpp

# -fPIC is required for every object that lands in the shared library, so it
# has to reach the bridge and platform objects too, not just ours.
SIM_CXXFLAGS := -O2 -fPIC -std=c++20 -I$(SRC_DIR) \
                -idirafter $(CONDA_PREFIX)/include -idirafter /usr/include
SIM_LDFLAGS := -shared -L$(CONDA_PREFIX)/lib -ldl -lz

# The platform layer's own library: XRT for vitis cards, VRT for the V80. VRT
# is built against the host glibc and libstdc++, so its platform layer is
# compiled by the host toolchain, and its headers need jsoncpp's and libxml2's.
ifeq ($(PLATFORM), vitis)
SIM_CXXFLAGS += -idirafter $(XILINX_XRT)/include
SIM_LDFLAGS += -L$(XILINX_XRT)/lib -luuid -lxrt_coreutil
else ifeq ($(PLATFORM), aved)
SIM_CXX := /usr/bin/g++
SIM_CXXFLAGS += $(shell pkg-config --cflags jsoncpp libxml-2.0)
SIM_LDFLAGS += -L/usr/lib/x86_64-linux-gnu -lvrt
endif

.PHONY: all lib clean check-gen

all: lib

check-gen:
	@test -f $(GEN_DIR)/FireSim-generated.const.h || { \
		echo "generated header not found under $(GEN_DIR)"; \
		echo "run 'make -C $(VORTEX_HOME)/hw/syn/firesim sources' then elaborate the target"; \
		exit 1; }

# The kernel build compiles these same driver sources as an executable, without
# -fPIC, into GEN_DIR. Reusing that directory would let make consider those
# objects current and link them into a shared library, which fails late with
# "final link failed: bad value". The position-independent build gets its own
# object directory, seeded with the generated headers it needs.
# Per platform: the objects embed that platform's generated widths, and make
# judges them by timestamp, so a directory shared between cards would link one
# card's objects into the other's library.
PIC_DIR := $(DESTDIR)/firesim_pic_$(PLATFORM)

$(PIC_DIR)/FireSim-generated.const.h: $(GEN_DIR)/FireSim-generated.const.h
	@mkdir -p $(PIC_DIR)
	@cp -f $(GEN_DIR)/FireSim-generated.const.* $(PIC_DIR)/ 2>/dev/null || true
	@cp -f $(GEN_DIR)/FireSim-generated.const.h $(PIC_DIR)/

ifeq ($(TARGET), avedsim)

# The metasimulator as the library: the FireSim-transformed target Verilated,
# with the emulated host interface (simif_emul) in place of the card's, linked
# into lib$(PROJECT).so so the runtime loads it exactly as it loads the card's.
# firesim_emul_main.cpp supplies the loop that --main would otherwise generate.
#
# The Vortex RTL comes from hw/syn/firesim's simulation flattening, the same
# sources the standalone metasim builds from; the generated FireSim RTL and
# headers come from the elaborated target.
SYN_DIR := $(ROOT_DIR)/hw/syn/firesim
SYN_BUILD_DIR ?= $(SYN_DIR)
SIM_SRCS_LIST := $(SYN_BUILD_DIR)/vortex_srcs.f
SIM_SRCS_DIR := $(SYN_BUILD_DIR)/vortex_srcs
ELAB_DIR := $(FIRESIM_PATH)/sim/generated-src/$(PLATFORM)/$(PLATFORM)-midasexamples-$(DESIGN)-$(TARGET_CONFIG)-$(PLATFORM_CONFIG)
EMUL_DIR := $(DESTDIR)/firesim_emul
EMUL_NAME := firesim_emul

EMUL_SRCS := $(SRC_DIR)/firesim_sim.cpp $(SRC_DIR)/firesim_emul_main.cpp

.PHONY: emul-sources
emul-sources:
	$(MAKE) -C $(SYN_DIR) sources SYN_BUILD_DIR=$(SYN_BUILD_DIR) FIRESIM_CARD=$(FIRESIM_CARD)

# Host toolchain, as for the aved platform library: the runtime that loads this
# is built with it. -fPIC reaches the Verilated objects too, through -CFLAGS.
# hw/syn/common.mk exports VERILATOR as a bare path and FireSim's `?=` would
# yield to it, so it is named here in full.
#
# The sub-make's own rule does not know the Vortex sources are inputs -- they
# arrive as flags -- so it would call the model current after a config change.
# This rule does know, so once it decides to rebuild, the old model goes.
$(EMUL_DIR)/V$(EMUL_NAME): $(EMUL_SRCS) $(SIM_SRCS_LIST) $(ELAB_DIR)/FireSim-generated.sv
	rm -f $@
	$(MAKE) -C $(SIMIF_DIR) verilator \
		PLATFORM=$(PLATFORM) \
		DRIVER_NAME=$(EMUL_NAME) \
		GEN_FILE_BASENAME=FireSim-generated \
		GEN_DIR=$(ELAB_DIR) \
		OUT_DIR=$(EMUL_DIR) \
		DRIVER="$(EMUL_SRCS)" \
		CXX=/usr/bin/g++ \
		VERILATOR="verilator --timing --cc --exe" \
		VERILATOR_FLAGS="-Wno-fatal -y $(SIM_SRCS_DIR) +libext+.sv+.v $$(tr '\n' ' ' < $(SIM_SRCS_LIST))" \
		CXXFLAGS="-O2 -fPIC -std=c++20 -DFIRESIM_EMUL -I$(SRC_DIR)" \
		LDFLAGS="-shared"

lib: emul-sources
	$(MAKE) $(EMUL_DIR)/V$(EMUL_NAME)
	cp -f $(EMUL_DIR)/V$(EMUL_NAME) $(DESTDIR)/lib$(PROJECT).so

else

# DRIVER_NAME is chosen so the sub-make emits lib$(PROJECT).so directly.
lib: check-gen $(PIC_DIR)/FireSim-generated.const.h
	$(MAKE) -C $(SIMIF_DIR) driver \
		MAIN=$(PLATFORM) PLATFORM=$(PLATFORM) \
		DRIVER_NAME=lib$(PROJECT).so.tmp \
		GEN_FILE_BASENAME=FireSim-generated \
		GEN_DIR=$(PIC_DIR) \
		OUT_DIR=$(DESTDIR) \
		DRIVER="$(DRIVER_SRCS)" \
		CXXFLAGS="$(SIM_CXXFLAGS)" \
		LDFLAGS="$(SIM_LDFLAGS)" \
		$(if $(SIM_CXX),CXX=$(SIM_CXX))
	mv -f $(DESTDIR)/lib$(PROJECT).so.tmp-$(PLATFORM) $(DESTDIR)/lib$(PROJECT).so

endif

clean:
	rm -f $(DESTDIR)/lib$(PROJECT).so
	rm -rf $(GEN_DIR)/bridges $(GEN_DIR)/core $(GEN_DIR)/*.o
