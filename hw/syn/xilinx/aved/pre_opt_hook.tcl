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
# What it does have is the opposite problem. pblock_slash constrains the RM to
# the dynamic region but spans all three SLRs, so it never asks the placer to
# keep the RM together: a 3%-utilization build was measured smeared over all
# three SLRs with 1691 SLL crossings, and all ten of its worst setup paths
# crossed SLR1->SLR2 (~1.9 ns on the crossing net alone, against a 3.333 ns
# period). xilinx_dfx_slr_confine.tcl pins the user logic to one SLR using
# ranges clipped from pblock_slash itself, so it constrains strictly inside the
# reconfigurable region rather than fighting it.

set tool_dir $::env(TOOL_DIR)
source ${tool_dir}/xilinx_async_bram_patch.tcl
source ${tool_dir}/xilinx_dfx_slr_confine.tcl
source ${tool_dir}/xilinx_noc_slr_steer.tcl

report_utilization -file hier_utilization.rpt -hierarchical -hierarchical_percentages
