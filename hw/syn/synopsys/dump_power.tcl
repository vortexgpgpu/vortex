## dump_power.tcl -- reopen a placed block, annotate power (SAIF or vectorless),
## and dump per-cell (x, y, total_power, area) to CSV for a spatial power-density grid.
## Pairs with powergrid.py. Run: xvfb-run -a fc_shell -f dump_power.tcl
## env: OUT_DIR (dir with <TOP>.dlib), SAIF_FILE, SAIF_INST, CSV, PDK_DBS
proc ev {n d} { if {[info exists ::env($n)] && [string trim $::env($n)] ne ""} { return $::env($n) }; return $d }
set OUT_DIR [ev OUT_DIR images]
set TOP     [ev TOP ""]
set SAIF    [ev SAIF_FILE ""]
set SI      [ev SAIF_INST ""]
set CSV     [ev CSV $OUT_DIR/cellpower.csv]
set DBS     [ev PDK_DBS ""]
if {$DBS eq ""} { set DBS [glob -nocomplain /mnt/nas0/eda.libs/asap7/asap7sc7p5t_28/LIB/NLDM/*_TT_nldm.db] }
set_app_var link_library [concat "*" $DBS]
if {$TOP ne "" && [file isdirectory $OUT_DIR/${TOP}.dlib]} {
  open_lib $OUT_DIR/${TOP}.dlib
} else {
  open_lib [lindex [glob $OUT_DIR/*.dlib] 0]
}
open_block [lindex [get_object_name [get_blocks]] 0]
link_block
if {$SAIF ne "" && [file exists $SAIF]} { if {[catch {read_saif $SAIF -strip_path $SI} e]} { puts "SAIFERR $e" } }
catch { report_power > $OUT_DIR/power.rpt }
set bb ""
foreach a {boundary_bbox bbox} { if {$bb eq ""} { catch { set bb [get_attribute [current_block] $a] } } }
set fh [open $CSV w]
puts $fh "#bbox [lindex [lindex $bb 0] 0] [lindex [lindex $bb 0] 1] [lindex [lindex $bb 1] 0] [lindex [lindex $bb 1] 1]"
puts $fh "x,y,power,area"
set cnt 0
foreach_in_collection c [get_cells -hierarchical -filter "is_hierarchical==false"] {
  set o [get_attribute $c origin]; set p [get_attribute $c total_power]; set ar [get_attribute $c area]
  if {$o eq "" || $p eq ""} continue
  puts $fh "[lindex $o 0],[lindex $o 1],$p,$ar"; incr cnt
}
close $fh
puts "DUMP DONE: $cnt cells -> $CSV"
exit
