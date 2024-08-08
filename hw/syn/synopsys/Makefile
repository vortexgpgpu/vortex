include ../common.mk

SCRIPT_DIR=./scripts

all: dc


#syn:
	#dc_shell-t -f esyn.tcl 2>&1 | tee vortex_syn.log
	#dc_shell -f esyn.tcl 2>&1 | tee vortex_syn.log
	#dc_shell -f $(SCRIPT_DIR)/dc/dc_script.tcl

dc:
	rm -rf rpt
	mkdir rpt
	dc_shell -f esyn.tcl 2>&1 | tee vortex_syn.log

clean:
	rm -f simv
	rm -f *.vcd
	rm -f *.key
	rm -rf csrc/
	rm -rf *.rpt
	rm -rf *.log
	rm -rf *.svf
	rm -rf *.ddc
	rm -rf results_synthesized.v
	rm -rf results_synthesized.sdc
	rm -rf alib-52/
	rm -rf rpt/
	rm -rf simv.daidir/
	rm -rf encounter*
	rm -rf ./synth_out