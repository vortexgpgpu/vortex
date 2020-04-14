PROJECT = Vortex
TOP_LEVEL_ENTITY = Vortex_SOC
SRC_FILE = ../Vortex.v
PROJECT_FILES = $(PROJECT).qpf $(PROJECT).qsf

QUARTUS_ROOT ?= /tools/reconfig/intel/18.0

# Part, Family
FAMILY = "Arria 10"
DEVICE = 10AX115N3F40E2SG   

# Executable Configuration
SYN_ARGS = --parallel --read_settings_files=on
FIT_ARGS = --part=$(DEVICE) --read_settings_files=on
ASM_ARGS =
STA_ARGS = --do_report_timing

# Build targets
all: $(PROJECT).sta.rpt $(PROJECT).pow.rpt

syn: $(PROJECT).syn.rpt

fit: $(PROJECT).fit.rpt

asm: $(PROJECT).asm.rpt

sta: $(PROJECT).sta.rpt

pow: $(PROJECT).pow.rpt

smart: smart.log

# Target implementations
STAMP = echo done >

$(PROJECT).syn.rpt: smart.log syn.chg $(SOURCE_FILES) 
	$(QUARTUS_ROOT)/quartus/bin/quartus_syn $(PROJECT) $(SYN_ARGS)
	$(STAMP) fit.chg

$(PROJECT).fit.rpt: smart.log fit.chg $(PROJECT).syn.rpt
	$(QUARTUS_ROOT)/quartus/bin/quartus_fit $(PROJECT) $(FIT_ARGS)
	$(STAMP) asm.chg
	$(STAMP) sta.chg

$(PROJECT).asm.rpt: smart.log asm.chg $(PROJECT).fit.rpt
	$(QUARTUS_ROOT)/quartus/bin/quartus_asm $(PROJECT) $(ASM_ARGS)

$(PROJECT).sta.rpt: smart.log sta.chg $(PROJECT).fit.rpt
	$(QUARTUS_ROOT)/quartus/bin/quartus_sta $(PROJECT)  $(STA_ARGS)
	$(QUARTUS_ROOT)/quartus/bin/quartus_sta -t VX_timing.tcl

$(PROJECT).pow.rpt: smart.log pow.chg $(PROJECT).fit.rpt
	$(QUARTUS_ROOT)/quartus/bin/quartus_pow $(PROJECT)

smart.log: $(PROJECT_FILES)
	$(QUARTUS_ROOT)/quartus/bin/quartus_sh --determine_smart_action $(PROJECT) > smart.log

# Project initialization
$(PROJECT_FILES):
	$(QUARTUS_ROOT)/quartus/bin/quartus_sh -t project.tcl -project $(PROJECT) -family $(FAMILY) -device $(DEVICE) -top $(TOP_LEVEL_ENTITY) -src $(SRC_FILE) -sdc vortex.sdc -inc "..;../interfaces;../pipe_regs;../cache;../VX_cache;../shared_memory;../compat"
	
syn.chg:
	$(STAMP) syn.chg

fit.chg:
	$(STAMP) fit.chg

sta.chg:
	$(STAMP) sta.chg

asm.chg:
	$(STAMP) asm.chg

pow.chg:
	$(STAMP) pow.chg

program: $(PROJECT).sof
	$(QUARTUS_ROOT)/quartus/bin/quartus_pgm --no_banner --mode=jtag -o "P;$(PROJECT).sof"
	
clean:
	rm -rf bin *.rpt *.chg *.qsf *.qpf smart.log *.htm *.eqn *.pin *.sof *.pof qdb incremental_db tmp-clearbox
