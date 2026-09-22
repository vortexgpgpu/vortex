# Runs on impl_1 immediately before opt_design, attached by the FireSim platform
# build through VORTEX_PRE_OPT_TCL.
#
# VX_async_ram_patch emits VX_placeholder black boxes wherever an
# asynchronous-read RAM has to be realized on a synchronous BRAM, and their
# contents are filled in here from the synthesized netlist. Without this,
# opt_design aborts with "DRC INBB-3 Black Box Instances" -- but only once a
# configuration is large enough to push those RAMs into BRAM.

set tool_dir $::env(TOOL_DIR)
source ${tool_dir}/xilinx_async_bram_patch.tcl
