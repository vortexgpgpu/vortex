## run_icc2.tcl
## Fusion Compiler placement + layout images for a synthesized netlist.
## Produces: placement, hierarchy-colored, cell-density heat map, and power-density
## heat map PNGs. Power uses SAIF activity if provided, else vectorless.
##
## NOTE: icc2_shell is not installed on this host; fc_shell shares the same ICC2 P&R
## engine / NDM database / physical libraries. Run headless via Xvfb:
##     xvfb-run -a fc_shell -f run_icc2.tcl
##
## Driven entirely by environment variables (see the Makefile 'images' target):
##   TOP            top module name                              (required)
##   NETLIST        gate-level netlist                            (default out/$TOP.mapped.v)
##   SDC            constraints                                   (default out/$TOP.post_compile.sdc)
##   OUT_DIR        output directory for the .dlib + PNGs         (default images)
##   SAIF_FILE      switching activity (optional -> vectorless)
##   SAIF_INST      instance scope of SAIF in the netlist         (optional)
##   PDK_TF         technology file (.tf)                         (required)
##   PDK_LEFS       cell/tech LEF files (space separated)         (required)
##   PDK_DBS        timing/power .db libs (space separated)       (required for power)
##   PDK_SRAM_DBS   SRAM .db libs (space separated)               (optional)
##   PDK_TLU        TLUPlus parasitic file                        (optional)
##   PDK_SCALE      internal length precision / scale factor      (default 10000)
##   UTIL           core utilization for the floorplan           (default 0.60)

proc ev {name def} {
  if {[info exists ::env($name)] && [string trim $::env($name)] ne ""} { return $::env($name) }
  return $def
}
proc note {m} { puts "\[run_icc2\] $m" }

set TOP        [ev TOP ""]
if {$TOP eq ""} { error "run_icc2: TOP not set" }
set NETLIST    [ev NETLIST  "out/${TOP}.mapped.v"]
set SDC        [ev SDC      "out/${TOP}.post_compile.sdc"]
set OUT_DIR    [ev OUT_DIR  "images"]
set SAIF_FILE  [ev SAIF_FILE ""]
set SAIF_INST  [ev SAIF_INST ""]
set TF         [ev PDK_TF   ""]
set LEFS       [ev PDK_LEFS ""]
set DBS        [ev PDK_DBS  ""]
set SRAM_DBS   [ev PDK_SRAM_DBS ""]
set TLU        [ev PDK_TLU  ""]
set SCALE      [ev PDK_SCALE 10000]
set UTIL       [ev UTIL     0.60]
set GDS_MODE   [ev GDS      0]
## GDS_PLACE_OPT=1 (default) uses timing-driven place_opt+CTS; =0 uses create_placement only
## (a safe fallback — place_opt can segfault optimizing large DesignWare datapath cells).
set GDS_PLACE_OPT [ev GDS_PLACE_OPT 1]
set route_done 0

if {![file exists $NETLIST]} { error "run_icc2: netlist not found: $NETLIST (run synthesis first)" }
if {$TF eq "" || $LEFS eq ""} { error "run_icc2: PDK_TF / PDK_LEFS not set (unsupported LIB_TYPE for layout?)" }
file mkdir $OUT_DIR
set_host_options -max_cores 8

## ---------------- 1. Library (timing + physical) ----------------
## The .db are 1x native and LEF/.tf may be Nx scaled; forcing -scale_factor avoids the
## LIB-055 precision clash. NDM cell libs are cached under ./CLIBs after the first run.
if {$DBS ne ""} { set_app_var link_library [concat "*" $DBS $SRAM_DBS] }
set DLIB $OUT_DIR/${TOP}.dlib
file delete -force $DLIB
catch { set_app_options -name lib.setting.use_tech_scale_factor -value true }
create_lib -technology $TF -ref_libs $LEFS -scale_factor $SCALE $DLIB

## ---------------- 2. Netlist ----------------
read_verilog -top $TOP $NETLIST
link_block

## ---------------- 3. Parasitics + constraints ----------------
if {$TLU ne "" && [file exists $TLU]} {
  set tlu_use $TLU
  ## ASAP7 ships asap07.tluplus gzip-compressed; decompress to OUT_DIR if needed.
  if {![catch {exec file -L $TLU} ft] && [string match -nocase "*gzip*" $ft]} {
    set tlu_use $OUT_DIR/parasitic.tluplus
    if {[catch {exec sh -c "gunzip -c [list $TLU] > [list $tlu_use]"} ge]} { note "WARN gunzip tlu: $ge"; set tlu_use $TLU }
  }
  if {[catch {read_parasitic_tech -tlup $tlu_use -name nominal} e]} { note "WARN parasitic: $e" } \
  else { catch { set_parasitic_parameters -late_spec nominal -early_spec nominal } }
}
if {[file exists $SDC]} { if {[catch {read_sdc $SDC} e]} { note "WARN sdc: $e" } }

## ---------------- 4. Floorplan + placement (+ optional route for GDS) ----------------
initialize_floorplan -core_utilization $UTIL -core_offset 5
if {[catch {connect_pg_net -automatic} e]} { note "WARN pg: $e" }
if {[catch {place_pins -self} e]}          { note "WARN pins: $e" }

if {$GDS_MODE == 1 && $DBS ne ""} {
  ## ---- full place-and-route to produce a routed GDS ----
  if {$GDS_PLACE_OPT == 1} {
    note "GDS mode: place_opt -> CTS -> route"
    if {[catch {place_opt} e]} { note "WARN place_opt: $e"; catch {create_placement}; catch {legalize_placement} }
    if {[catch {clock_opt} e]} { note "WARN clock_opt: $e" }
  } else {
    note "GDS mode (safe): create_placement -> route (skip place_opt/CTS)"
    if {[catch {create_placement} e]}   { note "WARN place: $e" }
    if {[catch {legalize_placement} e]} { note "WARN legalize: $e" }
  }
  if {[catch {route_auto} e]} { note "WARN route_auto: $e" } else { set route_done 1; note "routing complete" }
} else {
  if {[catch {create_placement} e]}   { note "WARN place: $e" }
  if {[catch {legalize_placement} e]} { note "WARN legalize: $e" }
}
catch { redirect $OUT_DIR/placement.rpt {report_placement} }
catch { save_block }

## ---------------- 5. Power (SAIF if given, else vectorless) ----------------
set have_power 0
if {$DBS ne ""} {
  if {$SAIF_FILE ne "" && [file exists $SAIF_FILE]} {
    note "power: SAIF activity from $SAIF_FILE"
    if {$SAIF_INST ne ""} {
      catch { read_saif $SAIF_FILE -strip_path $SAIF_INST }
    } else {
      catch { read_saif $SAIF_FILE }
    }
  } else {
    note "power: no SAIF -> vectorless estimation"
  }
  if {[catch {redirect $OUT_DIR/power.rpt {report_power}} e]} { note "WARN power: $e" } else { set have_power 1 }
} else {
  note "power: PDK_DBS not set -> skipping power map"
}

## ---------------- 5b. GDS ----------------
if {$GDS_MODE == 1} {
  if {[catch {write_gds $OUT_DIR/${TOP}.gds} e]} {
    note "WARN write_gds: $e"
  } else {
    note "wrote GDS ([expr {$route_done ? {routed} : {placed-only}}]): $OUT_DIR/${TOP}.gds"
  }
}

## ---------------- 6. Layout images ----------------
## FC's native GUI map modes (powerDensityMap/cellDensityMap/Hierarchy) do not compute
## in a headless/batch session (no interactive event loop). We instead render each figure
## by coloring leaf cells per bin with gui_change_highlight, which works in batch.
gui_start
set WIN [lindex [gui_get_window_ids] 0]
if {$WIN eq ""} { set WIN BlockWindow.1 }
set VIEW Layout.1
gui_set_active_window -window $WIN
catch { gui_zoom -window $WIN -full }

set HEAT    {blue cyan green yellow orange red}            ;# low -> high
set PALETTE {red green blue yellow cyan magenta orange white salmon}
set LEAF [get_cells -hierarchical -filter "is_hierarchical==false"]
note "leaf cells: [sizeof_collection $LEAF]"

proc snap {name} {
  global OUT_DIR WIN VIEW
  catch { gui_zoom -window $WIN -full }
  if {[catch {gui_write_window_image -window $VIEW -format png -file $OUT_DIR/${name}.png}]} {
    catch { gui_write_window_image -window $WIN -format png -file $OUT_DIR/${name}.png }
  }
}
proc clear_hl {} { global LEAF; catch { gui_change_highlight -remove -all_colors -collection $LEAF } }

## plain placement (no coloring)
clear_hl
snap placement

## A/C-style attribute heat map: QUANTILE bins (≈equal cells per color) by $attr.
## Quantile (rank-based) edges instead of linear, so a few high-power outliers don't
## collapse 99% of cells into the lowest bin — the spatial distribution stays legible.
proc heat_attr {attr name} {
  global LEAF HEAT OUT_DIR
  set s     [sort_collection $LEAF $attr]
  set names [get_object_name $s]
  set n     [llength $names]
  if {$n == 0} { note "$name: no leaf cells (skip)"; return 0 }
  set vmax [get_attribute [index_collection $s [expr {$n-1}]] $attr]
  if {$vmax eq "" || $vmax <= 0} { note "$name: no positive '$attr' (skip)"; return 0 }
  set nb [llength $HEAT]
  set lf [open $OUT_DIR/${name}_legend.txt w]
  puts $lf "# ${name}.png  -  '$attr' per leaf cell, QUANTILE bins (~equal cells/color), low(blue) -> high(red)"
  puts $lf "# color    ${attr}_range                       cells"
  clear_hl
  for {set i 0} {$i < $nb} {incr i} {
    set lo [expr {$n*$i/$nb}]; set hi [expr {$n*($i+1)/$nb}]
    if {$hi <= $lo} continue
    set sub   [get_cells [lrange $names $lo [expr {$hi-1}]]]
    set loval [get_attribute [index_collection $s $lo] $attr]
    set hival [get_attribute [index_collection $s [expr {$hi-1}]] $attr]
    set col   [lindex $HEAT $i]
    note [format "  %s %-6s \[%.4g, %.4g]: %d cells" $name $col $loval $hival [expr {$hi-$lo}]]
    puts $lf [format "%-8s \[%.4g, %.4g\]   %d" $col $loval $hival [expr {$hi-$lo}]]
    catch { gui_change_highlight -add -color $col -collection $sub }
  }
  close $lf
  snap $name
  clear_hl
  return 1
}

## B: color leaf cells by their hierarchical block (first 2 path levels; flat glue grouped)
proc color_hier {name} {
  global LEAF PALETTE OUT_DIR
  array unset grp
  foreach_in_collection c $LEAF {
    set f [get_object_name $c]
    set parts [split $f "/"]
    if {[llength $parts] <= 1} { set g "_top_glue_" } \
    elseif {[llength $parts] == 2} { set g [lindex $parts 0] } \
    else { set g "[lindex $parts 0]/[lindex $parts 1]" }
    lappend grp($g) $f
  }
  set lf [open $OUT_DIR/${name}_legend.txt w]
  puts $lf "# ${name}.png  -  color -> hierarchical block (leaf cells)."
  puts $lf "# Colors assigned by ALPHABETICAL block name from palette (cyclic; positional, not semantic):"
  puts $lf "#   $PALETTE"
  puts $lf "# color    block                                cells"
  clear_hl
  set i 0
  foreach g [lsort [array names grp]] {
    set col [lindex $PALETTE [expr {$i % [llength $PALETTE]}]]
    note "  hier $col <- $g ([llength $grp($g)] cells)"
    puts $lf [format "%-8s %-36s %d" $col $g [llength $grp($g)]]
    catch { gui_change_highlight -add -color $col -collection [get_cells $grp($g)] }
    incr i
  }
  close $lf
  snap $name
  clear_hl
}

## C: placement cell-density heat map. Grid the core by cell origin, color cells by the
## area-occupancy of their grid bin.
proc density_map {name {N 40}} {
  global LEAF HEAT OUT_DIR
  set bb ""
  foreach a {boundary_bbox bbox} { if {$bb eq ""} { catch { set bb [get_attribute [current_block] $a] } } }
  if {$bb eq ""} { note "$name: no block bbox (skip)"; return 0 }
  set x0 [lindex [lindex $bb 0] 0]; set y0 [lindex [lindex $bb 0] 1]
  set x1 [lindex [lindex $bb 1] 0]; set y1 [lindex [lindex $bb 1] 1]
  set dx [expr {($x1-$x0)/double($N)}]; set dy [expr {($y1-$y0)/double($N)}]
  if {$dx <= 0 || $dy <= 0} { note "$name: bad bbox (skip)"; return 0 }
  array unset barea; array unset bcells
  foreach_in_collection c $LEAF {
    set o [get_attribute $c origin]; set a [get_attribute $c area]
    if {$o eq "" || $a eq ""} continue
    set bx [expr {int(([lindex $o 0]-$x0)/$dx)}]; set by [expr {int(([lindex $o 1]-$y0)/$dy)}]
    if {$bx < 0} {set bx 0}; if {$bx >= $N} {set bx [expr {$N-1}]}
    if {$by < 0} {set by 0}; if {$by >= $N} {set by [expr {$N-1}]}
    set k "$bx.$by"
    set barea($k) [expr {([info exists barea($k)] ? $barea($k) : 0)+$a}]
    lappend bcells($k) [get_object_name $c]
  }
  set binA [expr {$dx*$dy}]; set dmax 0
  foreach k [array names barea] { set d [expr {$barea($k)/$binA}]; if {$d > $dmax} {set dmax $d} }
  if {$dmax <= 0} { note "$name: zero density (skip)"; return 0 }
  set n [llength $HEAT]; array unset lvl
  foreach k [array names barea] {
    set li [expr {int(($barea($k)/$binA)/$dmax*$n)}]; if {$li >= $n} {set li [expr {$n-1}]}
    foreach cn $bcells($k) { lappend lvl($li) $cn }
  }
  set lf [open $OUT_DIR/${name}_legend.txt w]
  puts $lf "# ${name}.png  -  cell-area occupancy per ${N}x${N} grid bin (fraction of bin area)"
  puts $lf "# gradient low(blue) -> high(red); dmax = [format %.3f $dmax]"
  puts $lf "# color    occupancy_range"
  clear_hl
  for {set i 0} {$i < $n} {incr i} {
    puts $lf [format "%-8s \[%.3f, %.3f)" [lindex $HEAT $i] [expr {$dmax*$i/$n}] [expr {$dmax*($i+1)/$n}]]
    if {[info exists lvl($i)]} { catch { gui_change_highlight -add -color [lindex $HEAT $i] -collection [get_cells $lvl($i)] } }
  }
  close $lf
  note "  density max = [format %.3f $dmax] (grid ${N}x${N})"
  snap $name
  clear_hl
  return 1
}

## B: hierarchy-colored floorplan
color_hier hierarchy

## C: cell-density heat map
density_map cell_density 40

## A: power heat map (vectorless or SAIF); fall back to leakage if no dynamic
if {$have_power} {
  if {![heat_attr total_power power_density]} { heat_attr leakage_power power_density }
} else {
  note "skipping power_density image (no power data)"
}

gui_stop
note "DONE. Images in $OUT_DIR/"
exit
