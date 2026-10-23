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

# xilinx_cone_probe.tcl <dcp> <out> <from> <to>...: for each from|to pair of cell
# patterns ("-" for any), up to 500 setup paths between them; prints the
# level histogram, the endpoints of the deepest and the deepest path cell by
# cell. Answers whether a cone the screen reports exists in Vivado's netlist,
# and how deep Vivado built it.
set dcp [lindex $argv 0]
set out [lindex $argv 1]
open_checkpoint $dcp
set f [open $out w]

proc cells_of {pat} {
  return [get_cells -quiet -hier -filter "IS_SEQUENTIAL && NAME =~ $pat"]
}

foreach {from to} [lrange $argv 2 end] {
  puts $f "=== from $from to $to"
  set args [list -quiet -setup -max_paths 500 -nworst 1]
  if {$from ne "-"} {
    set fc [cells_of $from]
    puts $f "from cells [llength $fc]"
    if {[llength $fc] == 0} continue
    lappend args -from $fc
  }
  if {$to ne "-"} {
    set tc [cells_of $to]
    puts $f "to cells [llength $tc]"
    if {[llength $tc] == 0} continue
    lappend args -to [get_pins -quiet -of_objects $tc -filter "DIRECTION == IN && IS_CLOCK == 0"]
  }
  set paths [get_timing_paths {*}$args]
  puts $f "paths [llength $paths]"
  if {[llength $paths] == 0} continue
  array unset hist
  set deepest ""
  set dl -1
  foreach p $paths {
    set l [get_property LOGIC_LEVELS $p]
    incr hist($l)
    if {$l > $dl} { set dl $l; set deepest $p }
  }
  set h ""
  foreach l [lsort -integer [array names hist]] { append h " $l:$hist($l)" }
  puts $f "levels$h"
  puts $f "deepest slack [get_property SLACK $deepest] levels $dl"
  puts $f "  from [get_property STARTPOINT_PIN $deepest]"
  puts $f "  to   [get_property ENDPOINT_PIN $deepest]"
  foreach c [get_cells -of_objects $deepest] {
    puts $f "    [get_property REF_NAME $c]  [get_property NAME $c]"
  }
}
close $f
