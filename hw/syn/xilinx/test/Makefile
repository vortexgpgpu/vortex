VIVADO = $(XILINX_VIVADO)/bin/vivado

RTL_DIR = ../../../rtl
AFU_DIR = $(RTL_DIR)/afu/xrt
SCRIPT_DIR = ../../../scripts

# include paths
FPU_INCLUDE = -I$(RTL_DIR)/fpu
TEX_INCLUDE = -I$(RTL_DIR)/tex
RASTER_INCLUDE = -I$(RTL_DIR)/raster
ROP_INCLUDE = -I$(RTL_DIR)/rop
RTL_INCLUDE = -I$(RTL_DIR) -I$(RTL_DIR)/libs -I$(RTL_DIR)/interfaces -I$(RTL_DIR)/core -I$(RTL_DIR)/cache
RTL_INCLUDE += $(FPU_INCLUDE) $(TEX_INCLUDE) $(RASTER_INCLUDE) $(ROP_INCLUDE)
RTL_INCLUDE += -Iproject_1_files

# compilation flags
CFLAGS += -DNDEBUG -DSYNTHESIS -DVIVADO
CFLAGS += $(CONFIGS)
CFLAGS += $(RTL_INCLUDE)
CFLAGS += -DEXT_F_DISABLE
#CFLAGS += -DEXT_GFX_ENABLE
#CFLAGS += -DNUM_CORES 4

# update memory layout for 2MB RAM
CFLAGS += -DSTARTUP_ADDR=32'h80000
CFLAGS += -DIO_BASE_ADDR=32'hFF000

COE_FILE := $(realpath project_1_files)/kernel.bin.coe
ESCAPED_COE_FILE := $(shell echo "$(COE_FILE)" | sed -e 's/[\/&]/\\&/g')

all: build

gen-sources: project_1/sources.txt
project_1/sources.txt:
	mkdir -p project_1
	$(SCRIPT_DIR)/gen_sources.sh $(CFLAGS) -P -EVX_fpu_fpnew.sv -Fproject_1/src -Oproject_1/sources.txt

project.tcl: project.tcl.in
	sed -e 's/%COE_FILE%/$(ESCAPED_COE_FILE)/g' < $< > $@

build: project_1/vortex.xpr
project_1/vortex.xpr: project_1/sources.txt project.tcl
	$(VIVADO) -mode batch -source project.tcl -tclargs project_1/sources.txt project_1_files $(SCRIPT_DIR)

run: project_1/vortex.xpr
	$(VIVADO) project_1/vortex.xpr &

clean:
	rm -rf project_1 project.tcl
