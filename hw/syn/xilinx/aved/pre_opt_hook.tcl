# Runs on impl_1 immediately before opt_design.
#
# VX_async_ram_patch emits VX_placeholder black boxes wherever an
# asynchronous-read RAM has to be realized on a synchronous BRAM. Their
# contents are filled in here, from the synthesized netlist. Without this,
# opt_design aborts with "DRC INBB-3 Black Box Instances ... has undefined
# contents" -- but only once a configuration is large enough to push those
# RAMs out of distributed LUTRAM and into BRAM, so small builds pass and
# large ones do not.
#
# Note: the xrt and dut flows also source xilinx_slr_pblocks.tcl here. That is
# deliberately omitted -- this AFU is implemented inside a DFX reconfigurable
# partition whose pblock already constrains placement, and nesting an SLR
# floorplan inside it would fight that constraint.

set tool_dir $::env(TOOL_DIR)
source ${tool_dir}/xilinx_async_bram_patch.tcl

report_utilization -file hier_utilization.rpt -hierarchical -hierarchical_percentages
