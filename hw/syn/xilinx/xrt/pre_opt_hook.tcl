set tool_dir $::env(TOOL_DIR)
source ${tool_dir}/xilinx_async_bram_patch.tcl
source ${tool_dir}/xilinx_slr_pblocks.tcl

report_utilization -file hier_utilization.rpt -hierarchical -hierarchical_percentages

# Implementation thread cap. Unset by default: the tools pick their own thread
# count. Set it to bound placer/router memory or to make a run reproducible --
# --vivado.impl.jobs sizes parallel runs, not the threads within one. This runs
# in the same Vivado session as place_design, so the setting carries into it.
if { [info exists ::env(IMPL_MAX_THREADS)] } {
  set_param general.maxThreads $::env(IMPL_MAX_THREADS)
  puts "INFO: \[VORTEX\] implementation threads capped at $::env(IMPL_MAX_THREADS)"
}

# Single-SLR kernel confinement. Unset by default: the SSI partitioner places
# the kernel. On a device this large the kernel occupies a few percent of one
# die, but the partitioner still spreads it across SLRs, so kernel-internal
# paths pay SLL crossing delay and inter-die clock skew -- both of which show up
# as routing delay on paths that carry almost no logic. Confining the kernel to
# one SLR removes that from every internal path; the platform's own regslices
# absorb the crossing to the shell and HBM.
if { [info exists ::env(VORTEX_SLR)] } {
  set vx_slr $::env(VORTEX_SLR)
  set vx_top [get_cells -quiet -hierarchical -filter {NAME =~ "*vortex_afu_1/inst" && IS_PRIMITIVE == 0}]
  if { [llength $vx_top] == 0 } {
    puts "CRITICAL WARNING: \[VORTEX\] kernel cell not found; cannot confine to ${vx_slr}"
  } else {
    set vx_top [lindex $vx_top 0]
    set vx_pb pblock_vortex_kernel
    if { [llength [get_pblocks -quiet $vx_pb]] == 0 } {
      create_pblock $vx_pb
    }
    add_cells_to_pblock $vx_pb $vx_top
    resize_pblock $vx_pb -add $vx_slr
    puts "INFO: \[VORTEX\] kernel [get_property NAME $vx_top] confined to ${vx_slr}"
  }
}

# Kernel clock constraint verification/repair.
# vpl's write_user_impl_clock_constraint derives the kernel generated-clock
# ratio from the clk_wizard solver; when the solver silently fails it emits a
# 1:1 copy of the MMCM reference input, so the whole implementation is timed
# at the reference frequency while the xclbin still programs the requested
# one at load. Re-derive the constraint from KERNEL_FREQ here and refuse to
# implement against a clock that does not match the target.
if { [info exists ::env(KERNEL_FREQ)] } {
  set vx_target_freq $::env(KERNEL_FREQ)
  set vx_target_period [expr {1000.0 / $vx_target_freq}]
  set vx_kclk [get_clocks -quiet clk_kernel_00_unbuffered_net]
  if { [llength $vx_kclk] == 0 } {
    puts "CRITICAL WARNING: \[VORTEX\] kernel clock 'clk_kernel_00_unbuffered_net' not found; cannot verify the ${vx_target_freq} MHz target on this platform"
  } else {
    set vx_period [get_property PERIOD $vx_kclk]
    if { [expr {abs($vx_period - $vx_target_period)}] > 0.001 } {
      puts "CRITICAL WARNING: \[VORTEX\] kernel clock constrained at ${vx_period} ns, target is ${vx_target_period} ns (${vx_target_freq} MHz); rebuilding the constraint"
      set vx_dst_pin [get_pins [get_property SOURCE_PINS $vx_kclk]]
      set vx_src_pin [get_pins [get_property SOURCE $vx_kclk]]
      set vx_ref_period [get_property PERIOD [get_clocks -of_objects $vx_src_pin]]
      set vx_ref_freq [expr {int(round(1000.0 / $vx_ref_period))}]
      set vx_freq_int [expr {int(round($vx_target_freq))}]
      set vx_a $vx_freq_int
      set vx_b $vx_ref_freq
      while { $vx_b } {
        set vx_t [expr {$vx_a % $vx_b}]
        set vx_a $vx_b
        set vx_b $vx_t
      }
      create_generated_clock -name clk_kernel_00_unbuffered_net \
        -multiply_by [expr {$vx_freq_int / $vx_a}] \
        -divide_by [expr {$vx_ref_freq / $vx_a}] \
        -source $vx_src_pin $vx_dst_pin
      set vx_period [get_property PERIOD [get_clocks clk_kernel_00_unbuffered_net]]
    }
    if { [expr {abs($vx_period - $vx_target_period)}] > 0.001 } {
      error "\[VORTEX\] kernel clock constraint is ${vx_period} ns but the target is ${vx_target_period} ns (${vx_target_freq} MHz); refusing to implement against a wrong kernel clock"
    }
    puts "INFO: \[VORTEX\] kernel clock constraint verified: ${vx_period} ns (${vx_target_freq} MHz)"
  }
}