//See LICENSE for license details.

package firesim.midasexamples

import chisel3._
import chisel3.util._

import org.chipsalliance.cde.config.{Config, Parameters}

import junctions._

import firesim.lib.bridges.{CompleteConfig, FASEDBridge, PeekPokeBridge, RationalClockBridge}
import firesim.lib.nasti._

/** Vortex's AXI memory port, flattened for a Chisel BlackBox.
  *
  * Vortex_axi declares its AXI channels as SystemVerilog unpacked arrays, which a BlackBox cannot express;
  * VX_firesim_wrap breaks them out per bank. Only the subset Vortex actually drives is exposed -- size, burst and
  * the protection/cache qualifiers are tied off inside the wrapper, so they are reconstructed on the Scala side.
  */
class VortexAxiIO(val addrBits: Int, val dataBits: Int, val idBits: Int) extends Bundle {
  val m_axi_mem_0_awvalid = Output(Bool())
  val m_axi_mem_0_awready = Input(Bool())
  val m_axi_mem_0_awaddr  = Output(UInt(addrBits.W))
  val m_axi_mem_0_awid    = Output(UInt(idBits.W))
  val m_axi_mem_0_awlen   = Output(UInt(8.W))

  val m_axi_mem_0_wvalid = Output(Bool())
  val m_axi_mem_0_wready = Input(Bool())
  val m_axi_mem_0_wdata  = Output(UInt(dataBits.W))
  val m_axi_mem_0_wstrb  = Output(UInt((dataBits / 8).W))
  val m_axi_mem_0_wlast  = Output(Bool())

  val m_axi_mem_0_bvalid = Input(Bool())
  val m_axi_mem_0_bready = Output(Bool())
  val m_axi_mem_0_bid    = Input(UInt(idBits.W))
  val m_axi_mem_0_bresp  = Input(UInt(2.W))

  val m_axi_mem_0_arvalid = Output(Bool())
  val m_axi_mem_0_arready = Input(Bool())
  val m_axi_mem_0_araddr  = Output(UInt(addrBits.W))
  val m_axi_mem_0_arid    = Output(UInt(idBits.W))
  val m_axi_mem_0_arlen   = Output(UInt(8.W))

  val m_axi_mem_0_rvalid = Input(Bool())
  val m_axi_mem_0_rready = Output(Bool())
  val m_axi_mem_0_rdata  = Input(UInt(dataBits.W))
  val m_axi_mem_0_rlast  = Input(Bool())
  val m_axi_mem_0_rid    = Input(UInt(idBits.W))
  val m_axi_mem_0_rresp  = Input(UInt(2.W))
}

/** The BlackBox's full port list.
  *
  * A BlackBox's field names have to match the Verilog port names exactly, and Chisel prefixes the fields of a nested
  * bundle with that bundle's name. The AXI ports are therefore inherited rather than nested, so they stay
  * `m_axi_mem_0_*` instead of becoming `mem_m_axi_mem_0_*`.
  */
class VortexWrapIO(addrBits: Int, dataBits: Int, idBits: Int, val dcrAddrBits: Int, val dcrDataBits: Int)
    extends VortexAxiIO(addrBits, dataBits, idBits) {
  val clk   = Input(Clock())
  val reset = Input(Bool())

  val dcr_req_valid = Input(Bool())
  val dcr_req_rw    = Input(Bool())
  val dcr_req_addr  = Input(UInt(dcrAddrBits.W))
  val dcr_req_data  = Input(UInt(dcrDataBits.W))

  val dcr_rsp_valid = Output(Bool())
  val dcr_rsp_data  = Output(UInt(dcrDataBits.W))

  val start = Input(Bool())
  val busy  = Output(Bool())
}

/** The Vortex GPU, as an opaque Verilog BlackBox.
  *
  * Golden Gate does not need to see inside: the FAME-1 transformation gates the clock at this boundary, which is
  * what makes the whole design host-decoupled. Vortex has no internal clock generation or gating, so a single
  * gated clock is sufficient.
  */
class VortexWrapBlackBox(val addrBits: Int, val dataBits: Int, val idBits: Int, val dcrAddrBits: Int, val dcrDataBits: Int)
    extends BlackBox
    with HasBlackBoxPath {
  override def desiredName = "VX_firesim_wrap"

  val io = IO(new VortexWrapIO(addrBits, dataBits, idBits, dcrAddrBits, dcrDataBits))

  // Only the packages and this wrapper are named explicitly; the rest of Vortex is found by library search over the
  // include directories, which the simulator build passes to Verilator. Handing over the whole tree instead does not
  // work: the modules use macro-computed width parameters, and once those files are pulled in as a flat list rather
  // than resolved as libraries, Verilator stops treating the widths as constants.
  private val srcList = sys.env.getOrElse(
    "VORTEX_RTL_SRCS",
    throw new RuntimeException("VORTEX_RTL_SRCS must point at the Vortex source list"),
  )
  scala.io.Source
    .fromFile(srcList)
    .getLines()
    .map(_.trim)
    .filter(l => l.nonEmpty && !l.startsWith("+"))
    .foreach(addPath)
}

/** Control surface the driver pokes over MMIO.
  *
  * Vortex's control plane is a handful of scalar signals rather than an AXI4-Lite block, so the PeekPoke bridge
  * drives it directly -- no custom control bridge is needed.
  */
class VortexCtrlIO(val dcrAddrBits: Int, val dcrDataBits: Int) extends Bundle {
  val dcr_req_valid = Input(Bool())
  val dcr_req_rw    = Input(Bool())
  val dcr_req_addr  = Input(UInt(dcrAddrBits.W))
  val dcr_req_data  = Input(UInt(dcrDataBits.W))
  val dcr_rsp_valid = Output(Bool())
  val dcr_rsp_data  = Output(UInt(dcrDataBits.W))
  val start         = Input(Bool())
  val busy          = Output(Bool())
}

class VortexDUT(nastiParams: NastiParameters)(implicit val p: Parameters) extends Module {
  val dcrAddrBits = 12
  val dcrDataBits = 32

  val io = IO(new Bundle {
    val nasti = new NastiIO(nastiParams)
    val ctrl  = new VortexCtrlIO(dcrAddrBits, dcrDataBits)
  })

  val vortex = Module(
    new VortexWrapBlackBox(
      addrBits    = nastiParams.addrBits,
      dataBits    = nastiParams.dataBits,
      idBits      = nastiParams.idBits,
      dcrAddrBits = dcrAddrBits,
      dcrDataBits = dcrDataBits,
    )
  )

  vortex.io.clk   := clock
  vortex.io.reset := reset.asBool

  private val mem = vortex.io

  // Vortex issues fixed-size INCR bursts at the full data-bus width; the wrapper does not carry the qualifiers, so
  // they are reconstructed here to form a complete AXI4 request for the memory model.
  private val fullSize = log2Ceil(nastiParams.dataBits / 8).U
  private val burstIncr = 1.U

  io.nasti.aw.valid      := mem.m_axi_mem_0_awvalid
  mem.m_axi_mem_0_awready := io.nasti.aw.ready
  io.nasti.aw.bits.addr  := mem.m_axi_mem_0_awaddr
  io.nasti.aw.bits.id    := mem.m_axi_mem_0_awid
  io.nasti.aw.bits.len   := mem.m_axi_mem_0_awlen
  io.nasti.aw.bits.size  := fullSize
  io.nasti.aw.bits.burst := burstIncr
  io.nasti.aw.bits.lock  := 0.U
  io.nasti.aw.bits.cache := 0.U
  io.nasti.aw.bits.prot  := 0.U
  io.nasti.aw.bits.qos    := 0.U
  io.nasti.aw.bits.region := 0.U
  io.nasti.aw.bits.user  := DontCare

  io.nasti.w.valid      := mem.m_axi_mem_0_wvalid
  mem.m_axi_mem_0_wready := io.nasti.w.ready
  io.nasti.w.bits.data  := mem.m_axi_mem_0_wdata
  io.nasti.w.bits.strb  := mem.m_axi_mem_0_wstrb
  io.nasti.w.bits.last  := mem.m_axi_mem_0_wlast
  io.nasti.w.bits.id    := DontCare
  io.nasti.w.bits.user  := DontCare

  mem.m_axi_mem_0_bvalid := io.nasti.b.valid
  io.nasti.b.ready       := mem.m_axi_mem_0_bready
  mem.m_axi_mem_0_bid    := io.nasti.b.bits.id
  mem.m_axi_mem_0_bresp  := io.nasti.b.bits.resp

  io.nasti.ar.valid      := mem.m_axi_mem_0_arvalid
  mem.m_axi_mem_0_arready := io.nasti.ar.ready
  io.nasti.ar.bits.addr  := mem.m_axi_mem_0_araddr
  io.nasti.ar.bits.id    := mem.m_axi_mem_0_arid
  io.nasti.ar.bits.len   := mem.m_axi_mem_0_arlen
  io.nasti.ar.bits.size  := fullSize
  io.nasti.ar.bits.burst := burstIncr
  io.nasti.ar.bits.lock  := 0.U
  io.nasti.ar.bits.cache := 0.U
  io.nasti.ar.bits.prot  := 0.U
  io.nasti.ar.bits.qos    := 0.U
  io.nasti.ar.bits.region := 0.U
  io.nasti.ar.bits.user  := DontCare

  mem.m_axi_mem_0_rvalid := io.nasti.r.valid
  io.nasti.r.ready       := mem.m_axi_mem_0_rready
  mem.m_axi_mem_0_rdata  := io.nasti.r.bits.data
  mem.m_axi_mem_0_rlast  := io.nasti.r.bits.last
  mem.m_axi_mem_0_rid    := io.nasti.r.bits.id
  mem.m_axi_mem_0_rresp  := io.nasti.r.bits.resp

  vortex.io.dcr_req_valid := io.ctrl.dcr_req_valid
  vortex.io.dcr_req_rw    := io.ctrl.dcr_req_rw
  vortex.io.dcr_req_addr  := io.ctrl.dcr_req_addr
  vortex.io.dcr_req_data  := io.ctrl.dcr_req_data
  vortex.io.start         := io.ctrl.start

  io.ctrl.dcr_rsp_valid := vortex.io.dcr_rsp_valid
  io.ctrl.dcr_rsp_data  := vortex.io.dcr_rsp_data
  io.ctrl.busy          := vortex.io.busy
}

/** Memory-port shape for a 32-bit Vortex build.
  *
  * These mirror the generated VX_config.vh rather than being chosen here: address width follows XLEN (32 for a
  * 32-bit build, 48 for 64-bit), data width is VX_CFG_PLATFORM_MEMORY_DATA_SIZE bytes, and the ID width is the
  * platform default. They must be regenerated alongside the RTL config, not edited independently.
  */
class VortexConfig
    extends Config((_, _, _) => { case NastiKey =>
      NastiParameters(dataBits = 512, addrBits = 32, idBits = 32)
    })

class VortexTarget(implicit val p: Parameters) extends RawModule {
  val clock = RationalClockBridge().io.clocks.head
  val reset = WireInit(false.B)

  withClockAndReset(clock, reset) {
    val vortex        = Module(new VortexDUT(p(NastiKey)))
    val fasedInstance = Module(new FASEDBridge(CompleteConfig(p(NastiKey))))
    fasedInstance.io.axi4  <> vortex.io.nasti
    fasedInstance.io.reset := reset
    fasedInstance.io.clock := clock
    PeekPokeBridge(clock, reset, ("ctrl", vortex.io.ctrl))
  }
}
