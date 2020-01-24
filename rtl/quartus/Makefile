PROJECT = Vortex
TOP_LEVEL_ENTITY = Vortex
SRC_FILE = Vortex.v
PROJECT_FILES = $(PROJECT).qpf $(PROJECT).qsf

QUARTUS_ROOT ?= /tools/reconfig/intel/18.0

# Part, Family
FAMILY = "Arria 10"
DEVICE = 10AX115N4F45I3SG   

# Executable Configuration
SYN_ARGS = --read_settings_files=on
FIT_ARGS = --part=$(DEVICE) --read_settings_files=on
ASM_ARGS =
STA_ARGS = --do_report_timing

# Build targets
all: smart.log $(PROJECT).asm.rpt $(PROJECT).sta.rpt

syn: smart.log $(PROJECT).syn.rpt

fit: smart.log $(PROJECT).fit.rpt

asm: smart.log $(PROJECT).asm.rpt

sta: smart.log $(PROJECT).sta.rpt

smart: smart.log

# Target implementations
STAMP = echo done >

$(PROJECT).syn.rpt: syn.chg $(SOURCE_FILES) 
	$(QUARTUS_ROOT)/quartus/bin/quartus_syn $(PROJECT) $(SYN_ARGS)
	$(STAMP) fit.chg

$(PROJECT).fit.rpt: fit.chg $(PROJECT).syn.rpt
	$(QUARTUS_ROOT)/quartus/bin/quartus_fit $(PROJECT) $(FIT_ARGS)
	$(STAMP) asm.chg
	$(STAMP) sta.chg

$(PROJECT).asm.rpt: asm.chg $(PROJECT).fit.rpt
	$(QUARTUS_ROOT)/quartus/bin/quartus_asm $(PROJECT) $(ASM_ARGS)

$(PROJECT).sta.rpt: sta.chg $(PROJECT).fit.rpt
	$(QUARTUS_ROOT)/quartus/bin/quartus_sta $(PROJECT)  $(STA_ARGS) 

smart.log: $(PROJECT_FILES)
	$(QUARTUS_ROOT)/quartus/bin/quartus_sh --determine_smart_action $(PROJECT) > smart.log

# Project initialization
$(PROJECT_FILES):
	$(QUARTUS_ROOT)/quartus/bin/quartus_sh -t project.tcl -project $(PROJECT) -family $(FAMILY) -device $(DEVICE) -top $(TOP_LEVEL_ENTITY) -src $(SRC_FILE) -sdc ../project.sdc
	
syn.chg:
	$(STAMP) syn.chg

fit.chg:
	$(STAMP) fit.chg

sta.chg:
	$(STAMP) sta.chg

asm.chg:
	$(STAMP) asm.chg

program: $(PROJECT).sof
	quartus_pgm --no_banner --mode=jtag -o "P;$(PROJECT).sof"
	
clean:
	rm -rf *.rpt *.chg *.qsf *.qpf smart.log *.htm *.eqn *.pin *.sof *.pof qdb incremental_db output_files tmp-clearbox
