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

# xilinx_cone_dump.tcl <dcp> <out> <pattern>...: for each pattern, the worst
# setup path into the matching cells, cell by cell, whatever its slack. When the
# path starts at a retimed register (_fret/_bret), the worst path into that
# register follows, up to four hops, and the levels are summed: the depth of
# the RTL cone as Vivado built it.
set dcp [lindex $argv 0]
set out [lindex $argv 1]
open_checkpoint $dcp
set f [open $out w]

proc dump_path {f p indent} {
  puts $f "${indent}slack [get_property SLACK $p] levels [get_property LOGIC_LEVELS $p] logic [get_property DATAPATH_LOGIC_DELAY $p] net [get_property DATAPATH_NET_DELAY $p]"
  puts $f "${indent}  from [get_property STARTPOINT_PIN $p]"
  puts $f "${indent}  to   [get_property ENDPOINT_PIN $p]"
  foreach c [get_cells -of_objects $p] {
    puts $f "${indent}    [get_property REF_NAME $c]  [get_property NAME $c]"
  }
}

foreach pat [lrange $argv 2 end] {
  puts $f "=== $pat"
  set cells [get_cells -quiet -hier -filter "IS_SEQUENTIAL && NAME =~ $pat"]
  if {[llength $cells] == 0} {
    set cells [get_cells -quiet -hier -filter "PRIMITIVE_LEVEL != MACRO && NAME =~ $pat"]
  }
  puts $f "cells [llength $cells]"
  if {[llength $cells] == 0} continue
  set pins [get_pins -quiet -of_objects $cells -filter "DIRECTION == IN && IS_CLOCK == 0"]
  set p [get_timing_paths -quiet -setup -max_paths 1 -nworst 1 -to $pins]
  if {[llength $p] == 0} { puts $f "no timed path"; continue }
  set total 0
  set indent ""
  for {set hop 0} {$hop < 5} {incr hop} {
    dump_path $f $p $indent
    set total [expr {$total + [get_property LOGIC_LEVELS $p]}]
    set start [get_cells -quiet -of_objects [get_property STARTPOINT_PIN $p]]
    if {$start eq "" || ![regexp {_[fb]ret} [get_property NAME $start]]} break
    set spins [get_pins -quiet -of_objects $start -filter "DIRECTION == IN && IS_CLOCK == 0"]
    set p [get_timing_paths -quiet -setup -max_paths 1 -nworst 1 -to $spins]
    if {[llength $p] == 0} break
    set indent "$indent  "
    puts $f "${indent}<- into retimed start [get_property NAME $start]"
  }
  puts $f "summed levels $total"
}
close $f
