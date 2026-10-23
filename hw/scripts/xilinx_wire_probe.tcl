# Copyright © 2019-2026
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

# xilinx_wire_probe.tcl <dcp> <out.csv> [max paths]: for the worst setup path of up
# to N end points, the placement the wire delay comes from: the Manhattan
# length of the path in tiles (sum over consecutive cells), the bounding box
# of its cells, and the largest fanout of a net on it. Feature 5's upper
# bound: how much of the net delay a perfect placement estimate explains.
set dcp [lindex $argv 0]
set out [lindex $argv 1]
set n [expr {[llength $argv] > 2 ? [lindex $argv 2] : 5000}]
open_checkpoint $dcp
set f [open $out w]
puts $f "slack,levels,logic,net,length,bbox,fanout,start,end"

proc tile_xy {c} {
  set site [get_sites -quiet -of_objects $c]
  if {$site eq ""} { return "" }
  set t [get_tiles -quiet -of_objects $site]
  if {$t eq ""} { return "" }
  return [list [get_property COLUMN $t] [get_property ROW $t]]
}

foreach p [get_timing_paths -setup -max_paths $n -nworst 1 -unique_pins] {
  set len 0
  set prev ""
  set xs {}
  set ys {}
  foreach c [get_cells -quiet -of_objects $p] {
    set xy [tile_xy $c]
    if {$xy eq ""} continue
    lappend xs [lindex $xy 0]
    lappend ys [lindex $xy 1]
    if {$prev ne ""} {
      set len [expr {$len + abs([lindex $xy 0] - [lindex $prev 0]) + abs([lindex $xy 1] - [lindex $prev 1])}]
    }
    set prev $xy
  }
  set bbox 0
  if {[llength $xs]} {
    set xs [lsort -integer $xs]
    set ys [lsort -integer $ys]
    set bbox [expr {[lindex $xs end] - [lindex $xs 0] + [lindex $ys end] - [lindex $ys 0]}]
  }
  set fo 0
  foreach net [get_nets -quiet -of_objects $p] {
    set k [get_property FLAT_PIN_COUNT $net]
    if {$k > $fo} { set fo $k }
  }
  puts $f "[get_property SLACK $p],[get_property LOGIC_LEVELS $p],[get_property DATAPATH_LOGIC_DELAY $p],[get_property DATAPATH_NET_DELAY $p],$len,$bbox,[expr {$fo - 1}],[get_property STARTPOINT_PIN $p],[get_property ENDPOINT_PIN $p]"
}
close $f
