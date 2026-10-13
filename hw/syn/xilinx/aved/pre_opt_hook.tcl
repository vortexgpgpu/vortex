# Runs on impl_1 immediately before opt_design.
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

report_utilization -file hier_utilization.rpt -hierarchical -hierarchical_percentages
