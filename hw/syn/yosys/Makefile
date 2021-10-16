PROJECT = Vortex
TOP_LEVEL_ENTITY = Vortex
SRC_FILE = Vortex.sv
RTL_DIR = ../../rtl

DEFINES = -DNDEBUG -DSYNTHESIS -DEXT_F_DISABLE -DNUM_CORES=1 -DNUM_THREADS=2 -DNUM_WARPS=2 -DMEM_BLOCK_SIZE=64

RTL_INCLUDE = -I$(RTL_DIR) -I$(RTL_DIR)/libs -I$(RTL_DIR)/interfaces -I$(RTL_DIR)/cache

# Build targets
all: build

output.v:
	./sv2v.sh $(DEFINES) $(RTL_INCLUDE) -ooutput.v

build: output.v	
	./synth.sh -t$(TOP_LEVEL_ENTITY) -soutput.v

clean:
	rm -rf output.v *.ys *.log
