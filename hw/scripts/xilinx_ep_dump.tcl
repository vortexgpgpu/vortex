# The end point table of a routed design: one row per setup end point under
# the horizon, `slack,levels,logic,net,start,end`, the format ci/beacon.py
# validate and calibrate read. The DUT flow writes one per build (ep.csv);
# standalone, it reads a checkpoint:
#
#   vivado -mode batch -source xilinx_ep_dump.tcl \
#          -tclargs <post_impl.dcp> <out.csv> [ref_period_ns] [horizon_ns]
#
# With ref_period_ns, slacks are moved to that clock period (slack minus the
# build's period minus ref_period_ns), so builds closed at different targets
# train one calibration. The horizon is in the reference period's slack.

proc write_ep_table {out {ref ""} {horizon 1.0}} {
  set period [get_property PERIOD [lindex [get_clocks] 0]]
  set shift  [expr {$ref eq "" ? 0.0 : $period - $ref}]
  set paths [get_timing_paths -quiet -setup -max_paths 1000000 -nworst 1 \
             -unique_pins -slack_lesser_than [expr {$horizon + $shift}]]
  set f [open $out w]
  puts $f "slack,levels,logic,net,start,end"
  foreach p $paths {
    set slack [get_property SLACK $p]
    if {$slack eq ""} { continue }
    puts $f [format "%.3f,%d,%.3f,%.3f,%s,%s" [expr {$slack - $shift}] \
      [get_property LOGIC_LEVELS $p] [get_property DATAPATH_LOGIC_DELAY $p] \
      [get_property DATAPATH_NET_DELAY $p] [get_property STARTPOINT_PIN $p] \
      [get_property ENDPOINT_PIN $p]]
  }
  close $f
  puts "INFO: $out: [llength $paths] end points, period $period ns, shift $shift ns"
}

if {[info exists ::argv] && [llength $::argv] >= 2
    && [file extension [lindex $::argv 0]] eq ".dcp"} {
  open_checkpoint [lindex $::argv 0]
  write_ep_table [lindex $::argv 1] {*}[lrange $::argv 2 3]
}
