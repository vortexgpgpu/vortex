PROJECT = Vortex
TOP_LEVEL_ENTITY = Vortex
SRC_FILE = Vortex.v
RTL_DIR = ../../rtl

DEFINES = -DNDEBUG -DSYNTHESIS -DEXT_F_DISABLE -DNUM_CORES=1 -DNUM_THREADS=2 -DNUM_WARPS=2

RTL_INCLUDE = -I$(RTL_DIR) -I$(RTL_DIR)/libs -I$(RTL_DIR)/interfaces -I$(RTL_DIR)/cache

# Build targets
all: build

build:
	./synth.sh -t$(TOP_LEVEL_ENTITY) -s$(SRC_FILE) $(DEFINES) $(RTL_INCLUDE)

clean:
	rm -rf sources.v *.ys *.log
