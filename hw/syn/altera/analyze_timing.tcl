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

set ProjectName [lindex $argv 0]
set SynMode     [lindex $argv 1]
if { $SynMode == "map" } {
	set FileSuffix "map"
} else {
	set FileSuffix "fit"
}

proc do_timing_checks { ProjectName FileSuffix } {
	# Validate timing DRC rules
	# REF: http://quartushelp.altera.com/14.0/mergedProjects/tafs/tafs/tcl_pkg_sta_ver_1.0_cmd_check_timing.htm
	check_timing -include {no_clock multiple_clock loops latches } -file $ProjectName.$FileSuffix.timing.check_errors.html
	# NOTE: metastability requires QSF setting of Synchronizer Identification = Auto
	# can also embed in Verilog: (* altera_attribute = "-name SYNCHRONIZER_IDENTIFICATION FORCED_IF_ASYNCHRONOUS" *) 
	report_metastability -nchains 100 -file $ProjectName.$FileSuffix.timing.check_metastability.html
}

proc do_timing_detailed_slackpaths { ProjectName FileSuffix SynMode } {
	# Detailed info for top 100 setup/hold paths
	if { $SynMode == "fit" } {
		set npaths_detailed 200
		set npaths_pairs    10000
		set npaths_maxslack 0.2
		# Create html reports showing details of each of the top 100 paths (creates html index + subdir with css/images/etc)
		set ExtraRTArgs "-show_routing"
		report_timing -setup    -nworst $npaths_detailed -detail full_path $ExtraRTArgs -file $ProjectName.$FileSuffix.timing.setup.html
		report_timing -hold     -nworst $npaths_detailed -detail full_path $ExtraRTArgs -file $ProjectName.$FileSuffix.timing.hold.html
		report_timing -recovery -nworst $npaths_detailed -detail full_path $ExtraRTArgs -file $ProjectName.$FileSuffix.timing.recovery.html
		report_timing -removal  -nworst $npaths_detailed -detail full_path $ExtraRTArgs -file $ProjectName.$FileSuffix.timing.removal.html

		# Create txt with (slack,src,dst) for cross-seed comparisons
		report_timing -setup    -nworst $npaths_pairs -less_than_slack $npaths_maxslack -detail summary -pairs_only -file $ProjectName.$FileSuffix.timing_paths.setup.txt
		report_timing -hold     -nworst $npaths_pairs -less_than_slack $npaths_maxslack -detail summary -pairs_only -file $ProjectName.$FileSuffix.timing_paths.hold.txt
		report_timing -recovery -nworst $npaths_pairs -less_than_slack $npaths_maxslack -detail summary -pairs_only -file $ProjectName.$FileSuffix.timing_paths.recovery.txt
		report_timing -removal  -nworst $npaths_pairs -less_than_slack $npaths_maxslack -detail summary -pairs_only -file $ProjectName.$FileSuffix.timing_paths.removal.txt

		# Histogram of setup/hold slacks across all clocks
		set allclocks [get_clocks]
		foreach_in_collection curclk $allclocks {
			set clkname [ get_clock_info -name $curclk ]
			create_slack_histogram -clock_name $clkname -setup -file $ProjectName.$FileSuffix.timing_histogram.$clkname.setup.html
			#create_slack_histogram -clock_name $clkname -hold  -file $ProjectName.$FileSuffix.timing_histogram.$clkname.hold.html
		}
	# Just emit simple setup paths if analyzing MAP netlist
	} else {
		set ExtraRTArgs ""
		report_timing -setup -nworst 100 -detail full_path $ExtraRTArgs -file $ProjectName.$FileSuffix.timing.setup.html
	}
}

proc do_timing_summary { ProjectName FileSuffix } {
	# Save summary into to single txt file
	create_timing_summary -setup         -file $ProjectName.$FileSuffix.timing.summary.txt
	create_timing_summary -hold  -append -file $ProjectName.$FileSuffix.timing.summary.txt
	report_clocks -summary       -append -file $ProjectName.$FileSuffix.timing.summary.txt
	report_clock_fmax_summary    -append -file $ProjectName.$FileSuffix.timing.summary.txt
}

proc do_timing_detailed_bottleneck_paths { ProjectName FileSuffix } {
	# Create bottleneck timing analysis with different metrics to analyze setup paths
	#proc custom_metric_fanins {arg} {
	#	upvar $arg metric
	#	set rating $metric(num_fanins)
	#	return $rating
	#}
	#report_bottleneck -cmetric custom_metric_fanins -file timing.bottlneck.num_fanins.html $tpaths
	set tpaths [ get_timing_paths -nworst 1000 -setup ]
	set tns_paths [ report_bottleneck -metric tns         $tpaths -stdout ]
	set np_paths  [ report_bottleneck -metric num_paths   $tpaths -stdout ] 
	set nfp_paths [ report_bottleneck -metric num_fpaths  $tpaths -stdout ]
	set nfo_paths [ report_bottleneck -metric num_fanouts $tpaths -stdout ]
	set nfi_paths [ report_bottleneck -metric num_fanins  $tpaths -stdout ]

	set fo [ open "$ProjectName.$FileSuffix.timing.setup.bottlenecks.txt" "w" ]
	puts $fo "Bottlenecks by TNS"
	puts $fo $tns_paths

	puts $fo "Bottlenecks by NumPaths"
	puts $fo $np_paths

	puts $fo "Bottlenecks by NumFailingPaths"
	puts $fo $nfp_paths

	puts $fo "Bottlenecks by NumFanOuts"
	puts $fo $nfo_paths

	puts $fo "Bottlenecks by NumFanIns"
	puts $fo $nfi_paths
}

# Iterate over all known operating conditions
# 3_H2_slow_850mv_100c / 3_H2_slow_850mv_100c / 3_H2_slow_850mv_0c / MIN_fast_850mv_100c / MIN_fast_850mv_0c
#foreach_in_collection oc [get_available_operating_conditions] {
#	set_operating_conditions $oc
#	post_message "Setting Operating Conditions $oc"
#	update_timing_netlist
#	report_timing -setup    -npaths 100 -file $ProjectName.timing.setup.html
#	report_timing -hold     -npaths 100 -file $ProjectName.timing.hold.html
#}

project_open $ProjectName

# => allows comparison of raw logic vs impact of routing delays
if { $SynMode == "map" } {
	create_timing_netlist -post_map
	read_sdc
	update_timing_netlist

	do_timing_detailed_slackpaths $ProjectName $FileSuffix $SynMode
	do_timing_summary $ProjectName $FileSuffix

	delete_timing_netlist

# normal post-par analysis (includes routing congestion/physical placement constraints)
} else {	
	create_timing_netlist
	read_sdc
	update_timing_netlist
	
	# Iterate over a single worst-case operating condition (grade/speed pre-selected based on netlist)
	set_operating_conditions -voltage 900 -temperature 100
	update_timing_netlist
	
	do_timing_checks $ProjectName $FileSuffix
	do_timing_detailed_slackpaths $ProjectName $FileSuffix $SynMode
	do_timing_detailed_bottleneck_paths $ProjectName $FileSuffix
	do_timing_summary $ProjectName $FileSuffix
	
	delete_timing_netlist
}

project_close