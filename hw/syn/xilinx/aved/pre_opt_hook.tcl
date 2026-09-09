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
# Note: the xrt and dut flows source xilinx_slr_pblocks.tcl here, which SPREADS
# a large design across SLRs to relieve congestion. That is deliberately still
# omitted -- this AFU sits in a DFX reconfigurable partition and does not have
# that problem.
#
# Nor does it need the opposite. Constraining the RM's placement was tried five
# ways and every one measured worse than leaving the placer alone; see
# pre_synth_hook.tcl for the numbers. The SLR split those attempts were
# chasing is fixed shell-side, in slash_base.tcl.

set tool_dir $::env(TOOL_DIR)
source ${tool_dir}/xilinx_async_bram_patch.tcl

report_utilization -file hier_utilization.rpt -hierarchical -hierarchical_percentages
