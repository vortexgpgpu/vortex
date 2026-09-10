ROOT_DIR := $(realpath ../..)
include $(ROOT_DIR)/config.mk

DESTDIR ?= $(CURDIR)
SRC_DIR := $(VORTEX_HOME)/sim/firesim

PROJECT := firesim

SIMIF_DIR := $(FIRESIM_PATH)/sim/midas/src/main/cc

PLATFORM ?= vitis
DESIGN ?= VortexTarget
TARGET_CONFIG ?= VortexConfig
PLATFORM_CONFIG ?= Default$(shell echo $(PLATFORM) | sed 's/^./\U&/')Config

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
                -idirafter $(CONDA_PREFIX)/include -idirafter /usr/include \
                -idirafter $(XILINX_XRT)/include
SIM_LDFLAGS := -shared -L$(CONDA_PREFIX)/lib -L$(XILINX_XRT)/lib \
               -luuid -lxrt_coreutil -ldl -lz

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
PIC_DIR := $(DESTDIR)/firesim_pic

$(PIC_DIR)/FireSim-generated.const.h: $(GEN_DIR)/FireSim-generated.const.h
	@mkdir -p $(PIC_DIR)
	@cp -f $(GEN_DIR)/FireSim-generated.const.* $(PIC_DIR)/ 2>/dev/null || true
	@cp -f $(GEN_DIR)/FireSim-generated.const.h $(PIC_DIR)/

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
		LDFLAGS="$(SIM_LDFLAGS)"
	mv -f $(DESTDIR)/lib$(PROJECT).so.tmp-$(PLATFORM) $(DESTDIR)/lib$(PROJECT).so

clean:
	rm -f $(DESTDIR)/lib$(PROJECT).so
	rm -rf $(GEN_DIR)/bridges $(GEN_DIR)/core $(GEN_DIR)/*.o
