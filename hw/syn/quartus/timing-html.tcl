package require cmdline

set options { 
    { "project.arg" "" "Project name" }
    { "outdir.arg" "timing-html" "Output directory" } 
}

array set opts [::cmdline::getoptions quartus(args) $options]

# Verify required parameters
set requiredParameters {project}
foreach p $requiredParameters {
    if {$opts($p) == ""} {
        puts stderr "Missing required parameter: -$p"
        exit 1
    }
}

project_open $opts(project)

set_global_assignment -name NUM_PARALLEL_PROCESSORS ALL

create_timing_netlist
read_sdc
update_timing_netlist

foreach_in_collection op [get_available_operating_conditions] {
  set_operating_conditions $op

  report_timing -setup -npaths 150 -detail full_path -multi_corner -pairs_only -nworst 8 \
    -file "$opts(outdir)/timing_paths_$op.html" \
    -panel_name "Critical paths for $op"

  create_slack_histogram -num_bins 50 -clock clk -multi_corner -file "$opts(outdir)/slack_histogram_$op.html"
}





