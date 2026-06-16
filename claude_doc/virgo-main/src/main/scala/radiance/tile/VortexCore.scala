// See LICENSE.Berkeley for license details.
// See LICENSE.SiFive for license details.

package radiance.tile

import chisel3._
import chisel3.util._
import chisel3.experimental._
import org.chipsalliance.cde.config.Parameters
import freechips.rocketchip.tile._
import radiance.subsystem.RadianceGemminiDataType

class VortexBundleA(
  tagWidth: Int,
  dataWidth: Int
) extends Bundle {
  assert(dataWidth % 8 == 0)
  val opcode = UInt(3.W) // FIXME: hardcoded
  val size = UInt(4.W) // FIXME: hardcoded
  val source = UInt(tagWidth.W) // FIXME: hardcoded
  val address = UInt(32.W) // FIXME: hardcoded
  val mask = UInt((dataWidth / 8).W) // FIXME: hardcoded
  val data = UInt(dataWidth.W) // FIXME: hardcoded
}

class VortexBundleD(
  tagWidth: Int,
  dataWidth: Int
) extends Bundle {
  assert(dataWidth % 8 == 0)
  val opcode = UInt(3.W) // FIXME: hardcoded
  val size = UInt(4.W) // FIXME: hardcoded
  val source = UInt(tagWidth.W) // FIXME: hardcoded
  val data = UInt(dataWidth.W) // FIXME: hardcoded
}

class VortexBundle(tile: RadianceTile)(implicit p: Parameters) extends CoreBundle {
  val clock = Input(Clock())
  val reset = Input(Reset())
  // val hartid = Input(UInt(tileIdLen.W))
  val reset_vector = Input(UInt(resetVectorLen.W))
  val interrupts = Input(new freechips.rocketchip.rocket.CoreInterrupts(false/*hasBeu*/))

  // conditionally instantiate ports depending on whether we want to use VX_cache or not
  // TODO: flatten this like dmem and smem
  val imem = if (!tile.radianceParams.useVxCache) Some(Vec(1, new Bundle {
    val a = Decoupled(new VortexBundleA(tagWidth = tile.imemTagWidth, dataWidth = 32))
    val d = Flipped(Decoupled(new VortexBundleD(tagWidth = tile.imemTagWidth, dataWidth = 32)))
  })) else None
  val mem = if (tile.radianceParams.useVxCache) Some(new Bundle {
    val a = Decoupled(new VortexBundleA(tagWidth = 15, dataWidth = 128))
    val d = Flipped(Decoupled(new VortexBundleD(tagWidth = 15, dataWidth = 128)))
    // val a = tile.memNode.out.head._1.a.cloneType
    // val d = Flipped(tile.memNode.out.head._1.d.cloneType)
  }) else None

  // Chisel doesn't support 2-D array in BlackBox interface to Verilog, so
  // everything needs to be 1-D flattened UInt with their widths configurable by numLSULanes.
  //
  // FIXME: hardcoded bitwidths
  val dmem_a_ready = Input(UInt((tile.numLsuLanes * 1).W))
  val dmem_a_valid = Output(UInt((tile.numLsuLanes * 1).W))
  val dmem_a_bits_opcode = Output(UInt((tile.numLsuLanes * 3).W))
  val dmem_a_bits_size = Output(UInt((tile.numLsuLanes * 4).W))
  val dmem_a_bits_source = Output(UInt((tile.numLsuLanes * tile.dmemTagWidth).W))
  val dmem_a_bits_address = Output(UInt((tile.numLsuLanes * 32).W))
  val dmem_a_bits_mask = Output(UInt((tile.numLsuLanes * 4).W))
  val dmem_a_bits_data = Output(UInt((tile.numLsuLanes * 32).W))

  val dmem_d_valid = Input(UInt((tile.numLsuLanes * 1).W))
  val dmem_d_bits_opcode = Input(UInt((tile.numLsuLanes * 3).W))
  val dmem_d_bits_size = Input(UInt((tile.numLsuLanes * 4).W))
  val dmem_d_bits_source = Input(UInt((tile.numLsuLanes * tile.dmemTagWidth).W))
  val dmem_d_bits_data = Input(UInt((tile.numLsuLanes * 32).W))
  val dmem_d_ready = Output(UInt((tile.numLsuLanes * 1).W))

  val smem_a_ready = Input(UInt((tile.numLsuLanes * 1).W))
  val smem_a_valid = Output(UInt((tile.numLsuLanes * 1).W))
  val smem_a_bits_opcode = Output(UInt((tile.numLsuLanes * 3).W))
  val smem_a_bits_size = Output(UInt((tile.numLsuLanes * 4).W))
  val smem_a_bits_source = Output(UInt((tile.numLsuLanes * tile.smemTagWidth).W))
  val smem_a_bits_address = Output(UInt((tile.numLsuLanes * 32).W))
  val smem_a_bits_mask = Output(UInt((tile.numLsuLanes * 4).W))
  val smem_a_bits_data = Output(UInt((tile.numLsuLanes * 32).W))

  val smem_d_valid = Input(UInt((tile.numLsuLanes * 1).W))
  val smem_d_bits_opcode = Input(UInt((tile.numLsuLanes * 3).W))
  val smem_d_bits_size = Input(UInt((tile.numLsuLanes * 4).W))
  val smem_d_bits_source = Input(UInt((tile.numLsuLanes * tile.smemTagWidth).W))
  val smem_d_bits_data = Input(UInt((tile.numLsuLanes * 32).W))
  val smem_d_ready = Output(UInt((tile.numLsuLanes * 1).W))

  val tc_a_valid = Output(UInt(2.W))
  val tc_a_bits_address = Output(UInt((2 * 32).W))
  val tc_a_bits_tag = Output(UInt((2 * 4).W))
  val tc_a_ready = Input(UInt(2.W))
  val tc_d_valid = Input(UInt(2.W))
  val tc_d_bits_data = Input(UInt((2 * 32 * 8).W))
  val tc_d_bits_tag = Input(UInt((2 * 4).W))
  val tc_d_ready = Output(UInt(2.W))

  // FIXME: hardcoded
  val barrierIdBits = tile.barrierMasterNode.out(0)._2.barrierIdBits
  val coreIdBits = tile.barrierMasterNode.out(0)._2.numCoreBits
  val gbar_req_valid = Output(Bool())
  val gbar_req_id = Output(UInt(barrierIdBits.W))
  val gbar_req_size_m1 = Output(UInt(coreIdBits.W))
  val gbar_req_core_id = Output(UInt(coreIdBits.W))
  val gbar_req_ready = Input(Bool())
  val gbar_rsp_valid = Input(Bool())
  val gbar_rsp_id = Input(UInt(barrierIdBits.W))

  val downstream_mem_busy = Input(Bool())

  val acc_read_in = Input(UInt(32.W))
  val acc_write_out = Output(UInt(32.W))
  val acc_write_en = Output(Bool())

  // val fpu = Flipped(new FPUCoreIO())
  //val rocc = Flipped(new RoCCCoreIO(nTotalRoCCCSRs))
  //val trace = Output(new TraceBundle)
  //val bpwatch = Output(Vec(coreParams.nBreakpoints, new BPWatch(coreParams.retireWidth)))
  val finished = Output(Bool())
  val wfi = Output(Bool())
  val traceStall = Input(Bool())
}

class Vortex(tile: RadianceTile)(implicit p: Parameters)
    extends BlackBox(
      // Each Vortex core gets tied-off core id of 0, 1, 2, 3, which is global
      // across multiple clusters.
      Map(
        "CORE_ID" -> tile.radianceParams.coreId,
        "TENSOR_FP16" -> (if (tile.radianceParams.core.tensorCoreFP16) 1 else 0),
        // TODO: can we get this as a parameter?
        "BOOTROM_HANG100" -> 0x10100,
        "NUM_THREADS" -> tile.numLsuLanes
      )
    )
    with HasBlackBoxResource with HasBlackBoxPath {
  // addResource("/vsrc/vortex/hw/unit_tests/generic_queue/testbench.v")
  // addResource("/vsrc/vortex/hw/unit_tests/VX_divide_tb.v")
  // addResource("/vsrc/vortex/hw/syn/synopsys/models/memory/cln28hpm/rf2_256x19_wm0/rf2_256x19_wm0_rtl.v")
  // addResource("/vsrc/vortex/hw/syn/synopsys/models/memory/cln28hpm/rf2_256x19_wm0/rf2_256x19_wm0.v")
  // addResource("/vsrc/vortex/hw/syn/synopsys/models/memory/cln28hpm/rf2_32x19_wm0/rf2_32x19_wm0.v")
  // addResource("/vsrc/vortex/hw/syn/synopsys/models/memory/cln28hpm/rf2_32x19_wm0/rf2_32x19_wm0_rtl.v")
  // addResource("/vsrc/vortex/hw/syn/synopsys/models/memory/cln28hpm/rf2_32x128_wm1/rf2_32x128_wm1_rtl.v")
  // addResource("/vsrc/vortex/hw/syn/synopsys/models/memory/cln28hpm/rf2_32x128_wm1/rf2_32x128_wm1.v")
  // addResource("/vsrc/vortex/hw/syn/synopsys/models/memory/cln28hpm/rf2_32x128_wm1/vsim/rf2_32x128_wm1_tb.v")
  // addResource("/vsrc/vortex/hw/syn/synopsys/models/memory/cln28hpm/rf2_256x128_wm1/rf2_256x128_wm1_rtl.v")
  // addResource("/vsrc/vortex/hw/syn/synopsys/models/memory/cln28hpm/rf2_256x128_wm1/rf2_256x128_wm1.v")
  // addResource("/vsrc/vortex/hw/syn/synopsys/models/memory/cln28hpm/rf2_128x128_wm1/rf2_128x128_wm1.v")
  // addResource("/vsrc/vortex/hw/syn/synopsys/models/memory/cln28hpm/rf2_128x128_wm1/rf2_128x128_wm1_rtl.v")
  // addResource("/vsrc/vortex/hw/syn/synopsys/models/memory/cln28hpc/rf2_32x128_wm1/rf2_32x128_wm1_rtl.v")
  // addResource("/vsrc/vortex/hw/syn/synopsys/models/memory/cln28hpc/rf2_32x128_wm1/rf2_32x128_wm1.v")
  // addResource("/vsrc/vortex/hw/syn/synopsys/models/memory/cln28hpc/rf2_32x128_wm1/vsim/rf2_32x128_wm1_tb.v")
  // addResource("/vsrc/vortex/hw/syn/modelsim/vortex_tb.v")

  addResource("/vsrc/vortex/hw/rtl/VX_gpu_pkg.sv")

  // addResource("/vsrc/vortex/hw/rtl/VX_cluster.sv")
  addResource("/vsrc/vortex/hw/rtl/VX_config.vh")
  addResource("/vsrc/vortex/hw/VX_config.h")
  addResource("/vsrc/vortex/hw/rtl/VX_define.vh")
  addResource("/vsrc/vortex/hw/rtl/VX_platform.vh")
  addResource("/vsrc/vortex/hw/rtl/VX_scope.vh")
  // addResource("/vsrc/vortex/hw/rtl/VX_socket.sv")
  addResource("/vsrc/vortex/hw/rtl/VX_types.vh")
  // addResource("/vsrc/vortex/hw/rtl/Vortex.sv")

  addResource("/vsrc/vortex/hw/rtl/core/VX_alu_unit.sv")
  addResource("/vsrc/vortex/hw/rtl/core/VX_commit.sv")
  addResource("/vsrc/vortex/hw/rtl/core/VX_core.sv")
  // These are top modules used for unittests
  // addResource("/vsrc/vortex/hw/rtl/core/VX_core_top.sv")
  // addResource("/vsrc/vortex/hw/rtl/cache/VX_cache_top.sv")
  // addResource("/vsrc/vortex/hw/rtl/cache/VX_cache_cluster_top.sv")
  addResource("/vsrc/vortex/hw/rtl/core/VX_csr_data.sv")
  addResource("/vsrc/vortex/hw/rtl/core/VX_csr_unit.sv")
  addResource("/vsrc/vortex/hw/rtl/core/VX_dcr_data.sv")
  addResource("/vsrc/vortex/hw/rtl/core/VX_decode.sv")
  addResource("/vsrc/vortex/hw/rtl/core/VX_dispatch.sv")
  addResource("/vsrc/vortex/hw/rtl/core/VX_dispatch_unit.sv")
  addResource("/vsrc/vortex/hw/rtl/core/VX_dispatch_unit_sane.sv")
  addResource("/vsrc/vortex/hw/rtl/core/VX_execute.sv")
  addResource("/vsrc/vortex/hw/rtl/core/VX_fetch.sv")
  addResource("/vsrc/vortex/hw/rtl/core/VX_gather_unit.sv")
  addResource("/vsrc/vortex/hw/rtl/core/VX_ibuffer.sv")
  addResource("/vsrc/vortex/hw/rtl/core/VX_int_unit.sv")
  addResource("/vsrc/vortex/hw/rtl/core/VX_ipdom_stack.sv")
  addResource("/vsrc/vortex/hw/rtl/core/VX_issue.sv")
  addResource("/vsrc/vortex/hw/rtl/core/VX_lsu_unit.sv")
  addResource("/vsrc/vortex/hw/rtl/core/VX_muldiv_unit.sv")
  addResource("/vsrc/vortex/hw/rtl/core/VX_operands.sv")
  addResource("/vsrc/vortex/hw/rtl/core/VX_operands_dup.sv")
  addResource("/vsrc/vortex/hw/rtl/core/VX_pending_instr.sv")
  addResource("/vsrc/vortex/hw/rtl/core/VX_schedule.sv")
  addResource("/vsrc/vortex/hw/rtl/core/VX_scoreboard.sv")
  addResource("/vsrc/vortex/hw/rtl/core/VX_sfu_unit.sv")
  addResource("/vsrc/vortex/hw/rtl/core/VX_smem_unit.sv")
  addResource("/vsrc/vortex/hw/rtl/core/VX_split_join.sv")
  addResource("/vsrc/vortex/hw/rtl/core/VX_trace.vh")
  addResource("/vsrc/vortex/hw/rtl/core/VX_wctl_unit.sv")

  // addResource("/vsrc/vortex/hw/rtl/cache/VX_cache_bank.sv")
  // addResource("/vsrc/vortex/hw/rtl/cache/VX_cache_bypass.sv")
  // need to disable this if VX_cache_cluster_top is disabled, otherwise causes
  // unconnected port error
  // addResource("/vsrc/vortex/hw/rtl/cache/VX_cache_cluster.sv")
  // addResource("/vsrc/vortex/hw/rtl/cache/VX_cache_data.sv")
  // addResource("/vsrc/vortex/hw/rtl/cache/VX_cache_define.vh")
  // addResource("/vsrc/vortex/hw/rtl/cache/VX_cache_init.sv")
  // addResource("/vsrc/vortex/hw/rtl/cache/VX_cache_mshr.sv")
  // addResource("/vsrc/vortex/hw/rtl/cache/VX_cache.sv")
  // addResource("/vsrc/vortex/hw/rtl/cache/VX_cache_tags.sv")
  // addResource("/vsrc/vortex/hw/rtl/cache/VX_cache_wrap.sv")

  // mem_arb is used in VX_socket or VX_cache_cluster
  // addResource("/vsrc/vortex/hw/rtl/mem/VX_mem_arb.sv")
  addResource("/vsrc/vortex/hw/rtl/mem/VX_mem_bus_if.sv")
  // addResource("/vsrc/vortex/hw/rtl/mem/VX_mem_perf_if.sv")
  addResource("/vsrc/vortex/hw/rtl/mem/VX_shared_mem.sv")
  addResource("/vsrc/vortex/hw/rtl/mem/VX_smem_switch.sv")

  // tex_unit missing in Vortex 2.0
  // addResource("/vsrc/vortex/hw/rtl/tex_unit/VX_tex_sat.sv")
  // addResource("/vsrc/vortex/hw/rtl/tex_unit/VX_tex_stride.sv")
  // addResource("/vsrc/vortex/hw/rtl/tex_unit/VX_tex_lerp.sv")
  // addResource("/vsrc/vortex/hw/rtl/tex_unit/VX_tex_addr.sv")
  // addResource("/vsrc/vortex/hw/rtl/tex_unit/VX_tex_mem.sv")
  // addResource("/vsrc/vortex/hw/rtl/tex_unit/VX_tex_format.sv")
  // addResource("/vsrc/vortex/hw/rtl/tex_unit/VX_tex_sampler.sv")
  // addResource("/vsrc/vortex/hw/rtl/tex_unit/VX_tex_unit.sv")
  // addResource("/vsrc/vortex/hw/rtl/tex_unit/VX_tex_define.vh")
  // addResource("/vsrc/vortex/hw/rtl/tex_unit/VX_tex_wrap.sv")

  // used when PERF_ENABLE is defined
  addResource("/vsrc/vortex/hw/rtl/mem/VX_mem_perf_if.sv")
  addResource("/vsrc/vortex/hw/rtl/interfaces/VX_pipeline_perf_if.sv")
  // used when GBAR_ENABLE is defined
  addResource("/vsrc/vortex/hw/rtl/mem/VX_gbar_bus_if.sv")
  // addResource("/vsrc/vortex/hw/rtl/mem/VX_gbar_arb.sv")
  // addResource("/vsrc/vortex/hw/rtl/mem/VX_gbar_unit.sv")

  addResource("/vsrc/vortex/hw/rtl/libs/VX_allocator.sv")
  // addResource("/vsrc/vortex/hw/rtl/libs/VX_avs_adapter.sv")
  // addResource("/vsrc/vortex/hw/rtl/libs/VX_axi_adapter.sv")
  addResource("/vsrc/vortex/hw/rtl/libs/VX_bits_insert.sv")
  addResource("/vsrc/vortex/hw/rtl/libs/VX_bits_remove.sv")
  // unused addResource("/vsrc/vortex/hw/rtl/libs/VX_bypass_buffer.sv")
  addResource("/vsrc/vortex/hw/rtl/libs/VX_cyclic_arbiter.sv")
  // unused addResource("/vsrc/vortex/hw/rtl/libs/VX_divider.sv")
  addResource("/vsrc/vortex/hw/rtl/libs/VX_dp_ram.sv")
  addResource("/vsrc/vortex/hw/rtl/libs/VX_elastic_adapter.sv")
  addResource("/vsrc/vortex/hw/rtl/libs/VX_elastic_buffer.sv")
  addResource("/vsrc/vortex/hw/rtl/libs/VX_fair_arbiter.sv")
  addResource("/vsrc/vortex/hw/rtl/libs/VX_fifo_queue.sv")
  addResource("/vsrc/vortex/hw/rtl/libs/VX_find_first.sv")
  addResource("/vsrc/vortex/hw/rtl/libs/VX_generic_arbiter.sv")
  addResource("/vsrc/vortex/hw/rtl/libs/VX_index_buffer.sv")
  // unused addResource("/vsrc/vortex/hw/rtl/libs/VX_index_queue.sv")
  addResource("/vsrc/vortex/hw/rtl/libs/VX_lzc.sv")
  addResource("/vsrc/vortex/hw/rtl/libs/VX_matrix_arbiter.sv")
  // addResource("/vsrc/vortex/hw/rtl/libs/VX_mem_adapter.sv")
  addResource("/vsrc/vortex/hw/rtl/libs/VX_mem_rsp_sel.sv")
  addResource("/vsrc/vortex/hw/rtl/libs/VX_mem_scheduler.sv")
  addResource("/vsrc/vortex/hw/rtl/libs/VX_multiplier.sv")
  // addResource("/vsrc/vortex/hw/rtl/libs/VX_mux.sv")
  addResource("/vsrc/vortex/hw/rtl/libs/VX_onehot_encoder.sv")
  // unused addResource("/vsrc/vortex/hw/rtl/libs/VX_onehot_mux.sv")
  addResource("/vsrc/vortex/hw/rtl/libs/VX_pending_size.sv")
  addResource("/vsrc/vortex/hw/rtl/libs/VX_pipe_register.sv")
  addResource("/vsrc/vortex/hw/rtl/libs/VX_pipe_buffer.sv")
  addResource("/vsrc/vortex/hw/rtl/libs/VX_toggle_buffer.sv")
  addResource("/vsrc/vortex/hw/rtl/libs/VX_stream_buffer.sv")
  addResource("/vsrc/vortex/hw/rtl/libs/VX_popcount.sv")
  addResource("/vsrc/vortex/hw/rtl/libs/VX_priority_arbiter.sv")
  addResource("/vsrc/vortex/hw/rtl/libs/VX_priority_encoder.sv")
  addResource("/vsrc/vortex/hw/rtl/libs/VX_reduce.sv")
  addResource("/vsrc/vortex/hw/rtl/libs/VX_reset_relay.sv")
  addResource("/vsrc/vortex/hw/rtl/libs/VX_rr_arbiter.sv")
  addResource("/vsrc/vortex/hw/rtl/libs/VX_scan.sv")
  // These VX_scope_* seems to be used for FPGA debugging; if we leave them in,
  // they cause elaboration errors
  // addResource("/vsrc/vortex/hw/rtl/libs/VX_scope_switch.sv")
  // addResource("/vsrc/vortex/hw/rtl/libs/VX_scope_tap.sv")
  addResource("/vsrc/vortex/hw/rtl/libs/VX_serial_div.sv")
  addResource("/vsrc/vortex/hw/rtl/libs/VX_serial_mul.sv")
  addResource("/vsrc/vortex/hw/rtl/libs/VX_shift_register.sv")
  addResource("/vsrc/vortex/hw/rtl/libs/VX_skid_buffer.sv")
  addResource("/vsrc/vortex/hw/rtl/libs/VX_sp_ram.sv")
  addResource("/vsrc/vortex/hw/rtl/libs/VX_stream_arb.sv")
  addResource("/vsrc/vortex/hw/rtl/libs/VX_stream_switch.sv")
  addResource("/vsrc/vortex/hw/rtl/libs/VX_stream_xbar.sv")

  // start comment out for synthesis ------------------------
  addResource("/vsrc/vortex/hw/dpi/float_dpi.cpp")
  addResource("/vsrc/vortex/hw/dpi/float_dpi.vh")
  addResource("/vsrc/vortex/hw/dpi/half.h")
  // end comment out for synthesis --------------------------
  addResource("/vsrc/vortex/hw/dpi/util_dpi.cpp")
  addResource("/vsrc/vortex/hw/dpi/util_dpi.vh")
  // needed dpi cpp files
  addResource("/vsrc/vortex/sim/common/bitmanip.h")
  addResource("/vsrc/vortex/sim/common/mem.cpp")
  addResource("/vsrc/vortex/sim/common/mem.h")
  addResource("/vsrc/vortex/sim/common/mempool.h")
  addResource("/vsrc/vortex/sim/common/rvfloats.cpp")
  addResource("/vsrc/vortex/sim/common/rvfloats.h")
  addResource("/vsrc/vortex/sim/common/simobject.h")
  addResource("/vsrc/vortex/sim/common/stringutil.h")
  addResource("/vsrc/vortex/sim/common/util.cpp")
  addResource("/vsrc/vortex/sim/common/util.h")
  addResource("/vsrc/vortex/sim/common/uuid_gen.h")

  // addResource("/csrc/softfloat_archive.a")
  addResource("/csrc/softfloat/include/internals.h")
  addResource("/csrc/softfloat/include/primitives.h")
  addResource("/csrc/softfloat/include/primitiveTypes.h")
  addResource("/csrc/softfloat/include/softfloat.h")
  addResource("/csrc/softfloat/include/softfloat_types.h")
  addResource("/csrc/softfloat/RISCV/specialize.h")

  // fpu
  // start comment out for synthesis ------------------------
  addResource("/vsrc/vortex/hw/rtl/fpu/VX_fpu_class.sv")
  addResource("/vsrc/vortex/hw/rtl/fpu/VX_fpu_cvt.sv")
  addResource("/vsrc/vortex/hw/rtl/fpu/VX_fpu_div.sv")
  addResource("/vsrc/vortex/hw/rtl/fpu/VX_fpu_dpi.sv")
  addResource("/vsrc/vortex/hw/rtl/fpu/VX_fpu_dsp.sv")
  addResource("/vsrc/vortex/hw/rtl/fpu/VX_fpu_fma.sv")
  addResource("/vsrc/vortex/hw/rtl/fpu/VX_fpu_ncomp.sv")
  addResource("/vsrc/vortex/hw/rtl/fpu/VX_fpu_rounding.sv")
  addResource("/vsrc/vortex/hw/rtl/fpu/VX_fpu_sqrt.sv")
  // end comment out for synthesis --------------------------
  addResource("/vsrc/vortex/hw/rtl/fpu/VX_fpu_to_csr_if.sv")
  addResource("/vsrc/vortex/hw/rtl/fpu/VX_fpu_define.vh")
  addResource("/vsrc/vortex/hw/rtl/fpu/VX_fpu_pkg.sv")
  addResource("/vsrc/vortex/hw/rtl/fpu/VX_fpu_fpnew.sv")
  addResource("/vsrc/vortex/hw/rtl/core/VX_fpu_unit.sv")

  // tensor core
  // this module is referenced from inside the Verilog RTL of the core
  // pipeline.
  // if (tile.radianceParams.core.tensorCoreFP16) {
  //   addResource("/vsrc/TensorDotProductUnit.sv")
  // } else {
  //   addResource("/vsrc/TensorDotProductUnitFP32.sv")
  // }

  // fpnew
  // compile order matters; package definitions (ex. fpnew_pkg) should be
  // compiled before all the other modules that reference them.  They are added
  // to vcs.mk
  addResource("/vsrc/vortex/third_party/fpnew/src/common_cells/include/common_cells/registers.svh")
  addResource("/vsrc/vortex/third_party/fpnew/src/fpnew_pkg.sv")
  addResource("/vsrc/vortex/third_party/fpnew/src/common_cells/src/cf_math_pkg.sv")
  addResource("/vsrc/vortex/third_party/fpnew/src/common_cells/src/cb_filter_pkg.sv")
  addResource("/vsrc/vortex/third_party/fpnew/src/common_cells/src/ecc_pkg.sv")
  addResource("/vsrc/vortex/third_party/fpnew/src/fpu_div_sqrt_mvp/hdl/defs_div_sqrt_mvp.sv")
  addResource("/vsrc/vortex/third_party/fpnew/src/fpnew_cast_multi.sv")
  addResource("/vsrc/vortex/third_party/fpnew/src/fpnew_classifier.sv")
  addResource("/vsrc/vortex/third_party/fpnew/src/fpnew_divsqrt_multi.sv")
  addResource("/vsrc/vortex/third_party/fpnew/src/fpnew_divsqrt_th_32.sv")
  addResource("/vsrc/vortex/third_party/fpnew/src/fpnew_fma_multi.sv")
  addResource("/vsrc/vortex/third_party/fpnew/src/fpnew_fma.sv")
  addResource("/vsrc/vortex/third_party/fpnew/src/fpnew_noncomp.sv")
  addResource("/vsrc/vortex/third_party/fpnew/src/fpnew_opgroup_block.sv")
  addResource("/vsrc/vortex/third_party/fpnew/src/fpnew_opgroup_fmt_slice.sv")
  addResource("/vsrc/vortex/third_party/fpnew/src/fpnew_opgroup_multifmt_slice.sv")
  addResource("/vsrc/vortex/third_party/fpnew/src/fpnew_rounding.sv")
  addResource("/vsrc/vortex/third_party/fpnew/src/fpnew_top.sv")

  addResource("/vsrc/vortex/third_party/fpnew/src/fpu_div_sqrt_mvp/hdl/control_mvp.sv")
  addResource("/vsrc/vortex/third_party/fpnew/src/fpu_div_sqrt_mvp/hdl/div_sqrt_mvp_wrapper.sv")
  addResource("/vsrc/vortex/third_party/fpnew/src/fpu_div_sqrt_mvp/hdl/div_sqrt_top_mvp.sv")
  addResource("/vsrc/vortex/third_party/fpnew/src/fpu_div_sqrt_mvp/hdl/iteration_div_sqrt_mvp.sv")
  addResource("/vsrc/vortex/third_party/fpnew/src/fpu_div_sqrt_mvp/hdl/norm_div_sqrt_mvp.sv")
  addResource("/vsrc/vortex/third_party/fpnew/src/fpu_div_sqrt_mvp/hdl/nrbd_nrsc_mvp.sv")
  addResource("/vsrc/vortex/third_party/fpnew/src/fpu_div_sqrt_mvp/hdl/preprocess_mvp.sv")

  // only include referenced modules in fpnew/common_cells; otherwise results
  // in elaboration error
  addResource("/vsrc/vortex/third_party/fpnew/src/common_cells/src/gray_to_binary.sv")
  addResource("/vsrc/vortex/third_party/fpnew/src/common_cells/src/lzc.sv")
  addResource("/vsrc/vortex/third_party/fpnew/src/common_cells/src/rr_arb_tree.sv")
  addResource("/vsrc/vortex/third_party/fpnew/src/common_cells/src/spill_register.sv")

  addResource("/vsrc/vortex/hw/rtl/interfaces/VX_branch_ctl_if.sv")
  addResource("/vsrc/vortex/hw/rtl/interfaces/VX_commit_csr_if.sv")
  addResource("/vsrc/vortex/hw/rtl/interfaces/VX_commit_if.sv")
  addResource("/vsrc/vortex/hw/rtl/interfaces/VX_commit_sched_if.sv")
  addResource("/vsrc/vortex/hw/rtl/interfaces/VX_dcr_bus_if.sv")
  addResource("/vsrc/vortex/hw/rtl/interfaces/VX_decode_if.sv")
  addResource("/vsrc/vortex/hw/rtl/interfaces/VX_decode_sched_if.sv")
  addResource("/vsrc/vortex/hw/rtl/interfaces/VX_dispatch_if.sv")
  addResource("/vsrc/vortex/hw/rtl/interfaces/VX_execute_if.sv")
  addResource("/vsrc/vortex/hw/rtl/interfaces/VX_fetch_if.sv")
  addResource("/vsrc/vortex/hw/rtl/interfaces/VX_ibuffer_if.sv")
  addResource("/vsrc/vortex/hw/rtl/interfaces/VX_operands_if.sv")
  addResource("/vsrc/vortex/hw/rtl/interfaces/VX_sched_csr_if.sv")
  addResource("/vsrc/vortex/hw/rtl/interfaces/VX_schedule_if.sv")
  addResource("/vsrc/vortex/hw/rtl/interfaces/VX_warp_ctl_if.sv")
  addResource("/vsrc/vortex/hw/rtl/interfaces/VX_writeback_if.sv")

  // tensor core
  addResource("/vsrc/vortex/hw/rtl/core/VX_tensor_core.sv")
  addResource("/vsrc/vortex/hw/rtl/mem/VX_tc_bus_if.sv")
  addResource("/vsrc/vortex/hw/rtl/mem/VX_tc_rf_if.sv")

  // hopper-style SMEM operand decoupling
  if (tile.radianceParams.core.tensorCoreDecoupled) {
    addResource("/vsrc/vortex/hw/rtl/core/VX_tensor_hopper_core.sv")
  //  addResource("/vsrc/vortex/hw/rtl/core/VX_tensor_ucode.vh")
    def addHopperTensorCore = {
      addPath("/scratch/hansung/chipyard/sims/vcs/generated-src/chipyard.unittest.TestHarness.TensorUnitTestConfig/gen-collateral/AddRawFN.sv")
      addPath("/scratch/hansung/chipyard/sims/vcs/generated-src/chipyard.unittest.TestHarness.TensorUnitTestConfig/gen-collateral/AddRecFN.sv")
      addPath("/scratch/hansung/chipyard/sims/vcs/generated-src/chipyard.unittest.TestHarness.TensorUnitTestConfig/gen-collateral/DotProductPipe.sv")
      addPath("/scratch/hansung/chipyard/sims/vcs/generated-src/chipyard.unittest.TestHarness.TensorUnitTestConfig/gen-collateral/FillBuffer_1.sv")
      addPath("/scratch/hansung/chipyard/sims/vcs/generated-src/chipyard.unittest.TestHarness.TensorUnitTestConfig/gen-collateral/FillBuffer.sv")
      addPath("/scratch/hansung/chipyard/sims/vcs/generated-src/chipyard.unittest.TestHarness.TensorUnitTestConfig/gen-collateral/metadataTable_4x5.sv")
      addPath("/scratch/hansung/chipyard/sims/vcs/generated-src/chipyard.unittest.TestHarness.TensorUnitTestConfig/gen-collateral/MulFullRawFN.sv")
      addPath("/scratch/hansung/chipyard/sims/vcs/generated-src/chipyard.unittest.TestHarness.TensorUnitTestConfig/gen-collateral/occupancyTable_4x1.sv")
      addPath("/scratch/hansung/chipyard/sims/vcs/generated-src/chipyard.unittest.TestHarness.TensorUnitTestConfig/gen-collateral/Queue1_TensorCoreDecoupled_Anon_1.sv")
      addPath("/scratch/hansung/chipyard/sims/vcs/generated-src/chipyard.unittest.TestHarness.TensorUnitTestConfig/gen-collateral/Queue1_TensorCoreDecoupled_Anon.sv")
      addPath("/scratch/hansung/chipyard/sims/vcs/generated-src/chipyard.unittest.TestHarness.TensorUnitTestConfig/gen-collateral/Queue1_TensorMemTag.sv")
      addPath("/scratch/hansung/chipyard/sims/vcs/generated-src/chipyard.unittest.TestHarness.TensorUnitTestConfig/gen-collateral/Queue4_TensorMemRespWithTag.sv")
      addPath("/scratch/hansung/chipyard/sims/vcs/generated-src/chipyard.unittest.TestHarness.TensorUnitTestConfig/gen-collateral/Queue5_TensorComputeTag.sv")
      addPath("/scratch/hansung/chipyard/sims/vcs/generated-src/chipyard.unittest.TestHarness.TensorUnitTestConfig/gen-collateral/ram_4x261.sv")
      addPath("/scratch/hansung/chipyard/sims/vcs/generated-src/chipyard.unittest.TestHarness.TensorUnitTestConfig/gen-collateral/ram_5x7.sv")
      addPath("/scratch/hansung/chipyard/sims/vcs/generated-src/chipyard.unittest.TestHarness.TensorUnitTestConfig/gen-collateral/RoundAnyRawFNToRecFN_ie8_is26_oe8_os24.sv")
      addPath("/scratch/hansung/chipyard/sims/vcs/generated-src/chipyard.unittest.TestHarness.TensorUnitTestConfig/gen-collateral/RoundAnyRawFNToRecFN_ie8_is47_oe8_os24.sv")
      addPath("/scratch/hansung/chipyard/sims/vcs/generated-src/chipyard.unittest.TestHarness.TensorUnitTestConfig/gen-collateral/RoundRawFNToRecFN_e8_s24.sv")
      addPath("/scratch/hansung/chipyard/sims/vcs/generated-src/chipyard.unittest.TestHarness.TensorUnitTestConfig/gen-collateral/SimpleTimer.sv")
      addPath("/scratch/hansung/chipyard/sims/vcs/generated-src/chipyard.unittest.TestHarness.TensorUnitTestConfig/gen-collateral/SourceGenerator.sv")
      addPath("/scratch/hansung/chipyard/sims/vcs/generated-src/chipyard.unittest.TestHarness.TensorUnitTestConfig/gen-collateral/StallingPipe_1.sv")
      addPath("/scratch/hansung/chipyard/sims/vcs/generated-src/chipyard.unittest.TestHarness.TensorUnitTestConfig/gen-collateral/StallingPipe_2.sv")
      addPath("/scratch/hansung/chipyard/sims/vcs/generated-src/chipyard.unittest.TestHarness.TensorUnitTestConfig/gen-collateral/StallingPipe.sv")
      addPath("/scratch/hansung/chipyard/sims/vcs/generated-src/chipyard.unittest.TestHarness.TensorUnitTestConfig/gen-collateral/TensorCoreDecoupled.sv")
      addPath("/scratch/hansung/chipyard/sims/vcs/generated-src/chipyard.unittest.TestHarness.TensorUnitTestConfig/gen-collateral/TensorDotProductUnit.sv")
    }
    // addHopperTensorCore
  }

  addResource("/vsrc/vortex/hw/rtl/core/VX_uop_sequencer.sv")
  addResource("/vsrc/vortex/hw/rtl/core/VX_reduce_unit.sv")
  addResource("/vsrc/vortex/hw/rtl/fpu/VX_tensor_dpu.sv")

  if (tile.radianceParams.useVxCache) {
    addResource("/vsrc/vortex/hw/rtl/libs/VX_pending_size.sv")
    addResource("/vsrc/vortex/hw/rtl/cache/VX_shared_mem.sv")
    addResource("/vsrc/vortex/hw/rtl/cache/VX_core_rsp_merge.sv")
    addResource("/vsrc/vortex/hw/rtl/cache/VX_tag_access.sv")
    addResource("/vsrc/vortex/hw/rtl/cache/VX_core_req_bank_sel.sv")
    addResource("/vsrc/vortex/hw/rtl/cache/VX_bank.sv")
    addResource("/vsrc/vortex/hw/rtl/cache/VX_data_access.sv")
    addResource("/vsrc/vortex/hw/rtl/cache/VX_flush_ctrl.sv")
    addResource("/vsrc/vortex/hw/rtl/cache/VX_nc_bypass.sv")
    addResource("/vsrc/vortex/hw/rtl/cache/VX_miss_resrv.sv")
    addResource("/vsrc/vortex/hw/rtl/cache/VX_cache.sv")
    addResource("/vsrc/vortex/hw/rtl/VX_mem_arb.sv")
    addResource("/vsrc/vortex/hw/rtl/VX_smem_arb.sv")
    addResource("/vsrc/vortex/hw/rtl/VX_mem_unit.sv")
    addResource("/vsrc/vortex/hw/rtl/VX_core.sv")
    addResource("/vsrc/vortex/hw/rtl/VX_core_wrapper.sv")
  } else {
    addResource("/vsrc/vortex/hw/rtl/VX_core_wrapper.sv")
  }

  val nTotalRoCCCSRs = 0
  val io = IO(new VortexBundle(tile))
}
