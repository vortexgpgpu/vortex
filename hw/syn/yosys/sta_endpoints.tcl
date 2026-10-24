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

# sta_endpoints.tcl: OpenSTA, the worst setup slack at the data input of
# every flip-flop, for comparing the logic screen per register with an
# ASIC flow (a CSV in the end point table format of ci/logic_screen.py).
# env: LIB NETLIST TOP SDC OUT
proc env {name} { return $::env($name) }
read_liberty [env LIB]
read_verilog [env NETLIST]
link_design [env TOP]
# The flow's SDC is in ns, whatever the library's time unit.
set_cmd_units -time ns -capacitance fF
read_sdc [env SDC]
set f [open [env OUT] w]
puts $f "slack,levels,logic,net,start,end"
# Flip-flops by library cell name (ASAP7: DFF*, SDF*); other cells have D
# pins too (OR4).
foreach pin [get_pins -hierarchical */D] {
  set cell [get_cells -of_objects $pin]
  if {![string match {[DS][DF]F*} [get_property $cell ref_name]]} continue
  set slack [get_property $pin slack_max]
  if {$slack eq "" || $slack eq "INF"} continue
  puts $f "$slack,0,0,0,-,[get_full_name $cell]/D"
}
close $f
