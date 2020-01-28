project_open Vortex

set_global_assignment -name NUM_PARALLEL_PROCESSORS ALL

create_timing_netlist
read_sdc
update_timing_netlist


foreach_in_collection op [get_available_operating_conditions] {
  set_operating_conditions $op

  report_timing -setup -npaths 20 -detail full_path -multi_corner \
    -file "bin/timing_paths_$op.html" \
    -panel_name "Critical paths for $op"

  create_slack_histogram -num_bins 50 -clock clk -multi_corner -file "bin/slack_histogram_$op.html"


}





