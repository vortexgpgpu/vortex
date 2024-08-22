ROOT_DIR := $(realpath ../../../../../..)
include $(ROOT_DIR)/config.mk

SRC_DIR := $(VORTEX_HOME)/hw/syn/altera/dut

RTL_DIR := $(VORTEX_HOME)/hw/rtl
AFU_DIR := $(RTL_DIR)/afu/opae
SCRIPT_DIR := $(VORTEX_HOME)/hw/scripts

IP_CACHE_DIR := $(ROOT_DIR)/hw/syn/altera/ip_cache/$(DEVICE_FAMILY)

ifeq ($(DEVICE_FAMILY), stratix10)
    FAMILY = "Stratix 10"
	DEVICE = 1SX280HN2F43E2VG
endif
ifeq ($(DEVICE_FAMILY), arria10)
    FAMILY = "Arria 10"
	DEVICE = 10AX115N3F40E2SG
endif

CONFIGS += -DNDEBUG
CONFIGS += -DQUARTUS
CONFIGS += -DSYNTHESIS

PROJECT_FILES = $(PROJECT).qpf $(PROJECT).qsf

# Executable Configuration
SYN_ARGS = --parallel --read_settings_files=on
FIT_ARGS = --parallel --part=$(DEVICE) --read_settings_files=on
ASM_ARGS =
STA_ARGS = --parallel --do_report_timing
POW_ARGS = --no_input_file --default_input_io_toggle_rate=60% --default_toggle_rate=20% --use_vectorless_estimation=off

# Build targets
all: gen-sources $(PROJECT).sta.rpt $(PROJECT).pow.rpt

gen-sources: src
src:
	mkdir -p src
	$(SCRIPT_DIR)/gen_sources.sh $(CONFIGS) $(RTL_INCLUDE) -T$(TOP_LEVEL_ENTITY) -P -Csrc

syn: $(PROJECT).syn.rpt

fit: $(PROJECT).fit.rpt

asm: $(PROJECT).asm.rpt

sta: $(PROJECT).sta.rpt

pow: $(PROJECT).pow.rpt

smart: smart.log

# Target implementations
STAMP = echo done >

$(PROJECT).syn.rpt: smart.log syn.chg
	quartus_syn $(SYN_ARGS) $(PROJECT)
	$(STAMP) fit.chg

$(PROJECT).fit.rpt: smart.log fit.chg $(PROJECT).syn.rpt
	quartus_fit $(FIT_ARGS) $(PROJECT)
	$(STAMP) asm.chg
	$(STAMP) sta.chg

$(PROJECT).asm.rpt: smart.log asm.chg $(PROJECT).fit.rpt
	quartus_asm $(ASM_ARGS) $(PROJECT)
	$(STAMP) pow.chg

$(PROJECT).sta.rpt: smart.log sta.chg $(PROJECT).fit.rpt
	quartus_sta $(STA_ARGS) $(PROJECT)

$(PROJECT).pow.rpt: smart.log pow.chg $(PROJECT).asm.rpt
	quartus_pow $(POW_ARGS) $(PROJECT)

smart.log: $(PROJECT_FILES)
	quartus_sh --determine_smart_action $(PROJECT) > smart.log

# Project initialization
$(PROJECT_FILES): gen-sources
	quartus_sh -t $(SRC_DIR)/project.tcl -project $(PROJECT) -family $(FAMILY) -device $(DEVICE) -top $(TOP_LEVEL_ENTITY) -src "$(SRC_FILE)" -sdc $(SRC_DIR)/project.sdc -inc "src"

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
	quartus_pgm --no_banner --mode=jtag -o "$(PROJECT).sof"

clean:
	rm -rf src bin *.rpt *.chg *.qsf *.qpf *.qws *.log *.htm *.eqn *.pin *.sof *.pof qdb incremental_db tmp-clearbox
