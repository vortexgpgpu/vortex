PROJECT = vortex_afu
TOP_LEVEL_ENTITY = vortex_afu
SRC_FILE = vortex_afu.sv
PROJECT_FILES = $(PROJECT).qpf $(PROJECT).qsf

# Part, Family
FAMILY = "Arria 10"
DEVICE = 10AX115N3F40E2SG   

# Executable Configuration
SYN_ARGS = --parallel --read_settings_files=on --set=VERILOG_MACRO=NOPAE=1
FIT_ARGS = --part=$(DEVICE) --read_settings_files=on
ASM_ARGS =
STA_ARGS = --do_report_timing

# Build targets
all: $(PROJECT).sta.rpt

syn: $(PROJECT).syn.rpt

fit: $(PROJECT).fit.rpt

asm: $(PROJECT).asm.rpt

sta: $(PROJECT).sta.rpt

smart: smart.log

# Target implementations
STAMP = echo done >

$(PROJECT).syn.rpt: smart.log syn.chg $(SOURCE_FILES) 
	quartus_syn $(PROJECT) $(SYN_ARGS)
	$(STAMP) fit.chg

$(PROJECT).fit.rpt: smart.log fit.chg $(PROJECT).syn.rpt
	quartus_fit $(PROJECT) $(FIT_ARGS)
	$(STAMP) asm.chg
	$(STAMP) sta.chg

$(PROJECT).asm.rpt: smart.log asm.chg $(PROJECT).fit.rpt
	quartus_asm $(PROJECT) $(ASM_ARGS)

$(PROJECT).sta.rpt: smart.log sta.chg $(PROJECT).fit.rpt
	quartus_sta $(PROJECT)  $(STA_ARGS) 

smart.log: $(PROJECT_FILES)
	quartus_sh --determine_smart_action $(PROJECT) > smart.log

# Project initialization
$(PROJECT_FILES):
	quartus_sh -t ../project.tcl -project $(PROJECT) -family $(FAMILY) -device $(DEVICE) -top $(TOP_LEVEL_ENTITY) -src $(SRC_FILE) -set "NOPAE" -sdc ../project.sdc -inc "../../../rtl;../../../rtl/libs;../../../rtl/interfaces;../../../rtl/pipe_regs;../../../rtl/cache;../../../opae;../../../opae/ccip"
	
syn.chg:
	$(STAMP) syn.chg

fit.chg:
	$(STAMP) fit.chg

sta.chg:
	$(STAMP) sta.chg

asm.chg:
	$(STAMP) asm.chg

program: $(PROJECT).sof
	quartus_pgm --no_banner --mode=jtag -o "$(PROJECT).sof"
	
clean:
	rm -rf bin *.rpt *.chg *.qsf *.qpf *.qws smart.log *.htm *.eqn *.pin *.sof *.pof qdb incremental_db tmp-clearbox
