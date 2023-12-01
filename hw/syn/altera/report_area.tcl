# Copyright © 2019-2023
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

load_package report

set ProjectName [lindex $argv 0]
set SynMode     [lindex $argv 1]

proc panel_to_csv { panel_name csv_file } {
    set fh [open $csv_file w]
    # Its possible for some panels to not exist based on design (ex. if no RAMs )
    set num_rows [get_number_of_rows -name $panel_name]
    catch {	    
	    for { set i 0 } { $i < $num_rows } { incr i } {
	        set row_data_raw [get_report_panel_row -name $panel_name -row $i]
	        set row_data [regsub -all , $row_data_raw ""]
	        puts $fh [join $row_data ","]
	    }
	}
    close $fh
}

# Dump names of all known panels
proc do_dump_panelnames { } {
	set fh [open "panels.txt" w]
	set panel_names [get_report_panel_names]
	foreach panel_name $panel_names {
		puts $fh "$panel_name"
	}
	close $fh
}

proc do_map_analysis { ProjectName } {
	# Save synthesis results
	set RSyn1 "Synthesis||Synthesis Source Files Read"
	set RSyn2 "Synthesis||Partition \"root_partition\"||Synthesis Resource Usage Summary for Partition \"root_partition\""
	set RSyn3 "Synthesis||Partition \"root_partition\"||Partition \"root_partition\" Resource Utilization by Entity"
	set RSyn4 "Synthesis||Partition \"root_partition\"||Synthesis RAM Summary for Partition \"root_partition\""
	set RSyn5 "Synthesis||Partition \"root_partition\"||Partition \"root_partition\" Optimization Results||Register Statistics||Registers Protected by Synthesis"
	set RSyn6 "Synthesis||Partition \"root_partition\"||Post-Synthesis Netlist Statistics for Partition \"root_partition\""
	panel_to_csv $RSyn1 "$ProjectName.syn.area.source_files.csv"
	panel_to_csv $RSyn2 "$ProjectName.syn.area.resource_summmary.csv"
	panel_to_csv $RSyn3 "$ProjectName.syn.area.resource_breakdown.csv"
	panel_to_csv $RSyn4 "$ProjectName.syn.area.ram_summary.csv"
	panel_to_csv $RSyn5 "$ProjectName.syn.area.regs_removed.csv"
	panel_to_csv $RSyn6 "$ProjectName.syn.area.stats.csv"
}

proc do_fit_analysis { ProjectName } {
	# Save par results
	set RPar1 "Fitter||Place Stage||Fitter Resource Usage Summary"
	set RPar2 "Fitter||Place Stage||Fitter Resource Utilization by Entity"
	set RPar3 "Fitter||Place Stage||Fitter Partition Statistics"
	set RPar4 "Fitter||Place Stage||Fitter RAM Summary"
	set RPar5 "Fitter||Plan Stage||Global & Other Fast Signals Summary"
	set RPar6 "Fitter||Place Stage||Non-Global High Fan-Out Signals"
	set RPar7 "Fitter||Route Stage||Routing Usage Summary"
	panel_to_csv $RPar1 "$ProjectName.fit.area.resource_summary.csv"
	panel_to_csv $RPar2 "$ProjectName.fit.area.resource_breakdown.csv"
	#panel_to_csv $RPar3 "$ProjectName.fit.area.stats.csv"
	panel_to_csv $RPar4 "$ProjectName.fit.area.ram_summary.csv"
	panel_to_csv $RPar5 "$ProjectName.fit.area.routing_global.csv"
	panel_to_csv $RPar6 "$ProjectName.fit.area.routing_high_fanout.csv"
	panel_to_csv $RPar7 "$ProjectName.fit.area.routing_summary.csv"
}

proc do_fit_analysis_timingsummary { ProjectName } {
	# Save timing results
	set RT1 "TimeQuest Timing Analyzer||Slow 900mV 100C Model||Slow 900mV 100C Model Fmax Summary"
	set RT2 "TimeQuest Timing Analyzer||Slow 900mV 100C Model||Slow 900mV 100C Model Setup Summary"
	set RT3 "TimeQuest Timing Analyzer||Slow 900mV 100C Model||Slow 900mV 100C Model Hold Summary"
	set RT4 "TimeQuest Timing Analyzer||Multicorner Timing Analysis Summary"
	panel_to_csv $RT1 "$ProjectName.fit.timing.summary.fmax.csv"
	panel_to_csv $RT2 "$ProjectName.fit.timing.summary.setup.csv"
	panel_to_csv $RT3 "$ProjectName.fit.timing.summary.hold.csv"
	panel_to_csv $RT4 "$ProjectName.fit.timing.summary.multicorner.csv"
}

project_open $ProjectName
load_report

# print available panels
#do_dump_panelnames

# => allows comparison of raw logic vs impact of routing delays
if { $SynMode == "map" } {
	do_map_analysis $ProjectName
# normal post-par analysis (includes routing congestion/physical placement constraints)
} else {	
	do_fit_analysis $ProjectName
	do_fit_analysis_timingsummary $ProjectName
}

unload_report
project_close