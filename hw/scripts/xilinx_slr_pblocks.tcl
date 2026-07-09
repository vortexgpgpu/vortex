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

# Per-SLR floorplan for Vortex on SSI (multi-SLR) Alveo/Versal devices.
#
# Enabled by default; disable with USE_SLR_PBLOCKS=0. Pins every large SLR-atomic
# block to a single SLR so the placer never splits one across an SLR boundary; the
# inter-SLR nets are carried only by the registered module-boundary seams (see the
# core in/out interface registration in docs/proposals/cache_out_reg_buffer_redesign.md).
# No-op on single-SLR devices. Device-queried (get_slrs), so it is
# portable across every SSI part in hw/syn/xilinx/xrt/platforms.mk — it round-
# robins across however many SLRs the device reports (2=U50, 3=U55C/U280/U200,
# 4=U250) and compiles out on monolithic parts (VCK5000/Versal, Zynq).
#
# Binning (agreed — see docs/proposals/slr_atomic_blocks.md, "Floorplan
# implication"):
#   anchor SLR (SLR0, next to HBM) : shared memory blocks — l2cache, l3cache and
#                                    the graphics caches (tcache/rcache/ocache/
#                                    rtcache), each treated like an L2 array.
#   round-robin across all SLRs    : one group per socket = the socket's compute
#                                    cores (g_cores[*].core, the agreed atom) plus
#                                    that socket's private L1 i/d-caches, kept
#                                    together so the tight, unregistered core<->L1
#                                    path never crosses an SLR; and each cluster-
#                                    level extension core (tex/raster/om/rtu +
#                                    dxa_core) as its own group.
#
# The atom is the CORE (g_cores[*].core): at SOCKET_SIZE=1 each socket holds one
# core, so a group is exactly {one core + its L1}; at SOCKET_SIZE>1 a socket's
# cores share one L1 over an unregistered link, so they are grouped together (the
# shared L1 cannot be in two SLRs at once). TCU and in-socket tex/rtu ride inside
# their core and need no separate bin.

if {![info exists ::env(USE_SLR_PBLOCKS)] || $::env(USE_SLR_PBLOCKS) ne "0"} {
    set slrs [lsort [get_slrs]]
    set nslr [llength $slrs]
    if {$nslr < 2} {
        puts "SLR-PBLOCKS: device has $nslr SLR(s); floorplan skipped"
    } else {
        # Collect cells matching any of a list of hierarchical regexps. -quiet
        # keeps config-dependent patterns that match nothing from erroring.
        proc slr_collect {patterns} {
            set out {}
            foreach p $patterns {
                foreach c [get_cells -quiet -hierarchical -regexp $p] { lappend out $c }
            }
            return $out
        }
        # Escape regex metacharacters in a literal hierarchical instance name so
        # it can be used as an anchored prefix in a child get_cells query.
        proc slr_re_escape {s} {
            return [string map {\\ \\\\ . \\. \[ \\\[ \] \\\] ( \\( ) \\) + \\+ * \\* ? \\? ^ \\^ $ \\$ | \\| \{ \\\{ \} \\\}} $s]
        }

        # ---- shared blocks -> anchor SLR (HBM side) ----
        # Shared caches (l2/l3 + graphics caches) and the shared control units
        # (KMU dispatch, global-barrier) that fan out to every core. Their
        # crossings to the cores are register-bounded by the kmu/gbar arbs, so
        # the units themselves live in the anchor SLR next to L2/L3.
        set caches [slr_collect [list \
            {.*/l3cache} {.*/l2cache} \
            {.*/tcache} {.*/rcache} {.*/ocache} {.*/rtcache} \
            {.*/kmu} {.*/gbar_unit} \
        ]]

        # ---- compute atoms -> round-robin, each pinned whole ----
        # One group per socket: its cores (g_cores[*].core) + its private L1s.
        set groups {}
        set glabels {}
        foreach s [get_cells -quiet -hierarchical -regexp {.*/g_sockets\[[0-9]+\]\.socket}] {
            set es [slr_re_escape $s]
            set g {}
            foreach c [get_cells -quiet -hierarchical -regexp "${es}/g_cores\\\[\[0-9\]+\\\]\\.core"] { lappend g $c }
            foreach lc [get_cells -quiet -hierarchical -regexp "${es}/(icache|dcache)"]           { lappend g $lc }
            if {[llength $g]} { lappend groups $g; lappend glabels "socket [get_property NAME $s]" }
        }
        # Each cluster-level extension core is its own group.
        foreach e [slr_collect [list \
            {.*\.tex_core} \
            {.*\.raster_core} \
            {.*\.om_core} \
            {.*\.rtu_core} \
            {.*/dxa_core} \
        ]] {
            lappend groups [list $e]; lappend glabels [get_property NAME $e]
        }

        puts "SLR-PBLOCKS: $nslr SLRs ($slrs); [llength $caches] shared-cache atom(s), [llength $groups] compute group(s)"

        if {[llength $groups] == 0 && [llength $caches] == 0} {
            puts "SLR-PBLOCKS: WARNING no target instances matched; floorplan not applied"
        } else {
            # anchor SLR0: shared caches + cluster glue.
            create_pblock pb_slr0
            resize_pblock pb_slr0 -add [lindex $slrs 0]
            if {[llength $caches]} {
                add_cells_to_pblock pb_slr0 $caches
                foreach c $caches { puts "SLR-PBLOCKS: [get_property NAME $c] -> [lindex $slrs 0] (cache)" }
            }
            # compute groups: round-robin each to its own single-SLR pblock.
            for {set i 0} {$i < [llength $groups]} {incr i} {
                set slr [lindex $slrs [expr {$i % $nslr}]]
                set pb "pb_atom$i"
                create_pblock $pb
                resize_pblock $pb -add $slr
                add_cells_to_pblock $pb [lindex $groups $i]
                puts "SLR-PBLOCKS: [lindex $glabels $i] ([llength [lindex $groups $i]] cell(s)) -> $slr"
            }
        }
    }
}
