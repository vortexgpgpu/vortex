package radiance.core

import chisel3._
import chisel3.util._
import org.chipsalliance.cde.config.{Field, Parameters}
import org.chipsalliance.diplomacy.lazymodule.{LazyModule, LazyModuleImp}
import freechips.rocketchip.tilelink._
import freechips.rocketchip.diplomacy.{IdRange, AddressSet, BufferParams}
import radiance.memory.{SourceGenerator, TraceLine, TLPrintf}

case class SIMTCoreParams(
    nWarps: Int = 4,     // # of warps in the core
    nCoreLanes: Int = 4, // # of SIMT threads in the core
    nMemLanes: Int = 4,  // # of memory lanes in the memory interface to the
                         // cache; relates to the LSU lanes
    nSrcIds: Int = 8     // # of source IDs allocated to each of the nMemLanes
)
case class MemtraceCoreParams(
    tracefilename: String = "undefined",
    traceHasSource: Boolean = false
)

case object SIMTCoreKey extends Field[Option[SIMTCoreParams]](None /*default*/ )
case object MemtraceCoreKey
    extends Field[Option[MemtraceCoreParams]](None /*default*/ )

// #############################################################################
// FIXME: copy-paste from MemFuzzer
// #############################################################################

class Emulator(
    numLanes: Int,
    numSrcIds: Int,
    wordSizeInBytes: Int,
)(implicit p: Parameters)
    extends LazyModule {
  val laneNodes = Seq.tabulate(numLanes) { i =>
    val clientParam = Seq(
      TLMasterParameters.v1(
        name = "Emulator" + i.toString,
        sourceId = IdRange(0, numSrcIds)
        // visibility = Seq(AddressSet(0x0000, 0xffffff))
      )
    )
    TLClientNode(Seq(TLMasterPortParameters.v1(clientParam)))
  }

  val node = TLIdentityNode()
  laneNodes.foreach(node := _)

  lazy val module = new EmulatorImp(this, numLanes, numSrcIds, wordSizeInBytes)
}

class EmulatorImp(
    outer: Emulator,
    numLanes : Int,
    numSrcIds: Int,
    wordSizeInBytes: Int,
) extends LazyModuleImp(outer) {
  val io = IO(new Bundle {
    val finished = Output(Bool())
  })
  val sim = Module(new SimEmulator(numLanes))

  sim.io.clock := clock
  sim.io.reset := reset.asBool

  sim.io.a.ready := VecInit(outer.laneNodes.map { node =>
    val (tlOut, _) = node.out(0)
    tlOut.a.ready
  }).asUInt

  io.finished := sim.io.finished

  // connect Verilog <-> Chisel IO
  // Verilog IO flattened across all lanes
  val laneReqs = Wire(Vec(numLanes, Decoupled(new TraceLine)))
  val addrW = laneReqs(0).bits.address.getWidth
  val sizeW = laneReqs(0).bits.size.getWidth
  val dataW = laneReqs(0).bits.data.getWidth
  laneReqs.zipWithIndex.foreach { case (req, i) =>
    req.valid := sim.io.a.valid(i)
    req.bits.source := 0.U // DPI doesn't generate contain source id
    req.bits.address := sim.io.a.address(addrW * (i + 1) - 1, addrW * i)
    req.bits.is_store := sim.io.a.is_store(i)
    req.bits.size := sim.io.a.size(sizeW * (i + 1) - 1, sizeW * i)
    req.bits.data := sim.io.a.data(dataW * (i + 1) - 1, dataW * i)
  }
  sim.io.a.ready := VecInit(laneReqs.map(_.ready)).asUInt

  val laneResps = Wire(Vec(numLanes, Flipped(Decoupled(new TraceLine))))
  laneResps.zipWithIndex.foreach { case (resp, i) =>
    resp.ready := sim.io.d.ready(i)
    // TODO: not handled in DPI
    resp.bits.source := DontCare
    resp.bits.address := DontCare
    resp.bits.data := DontCare
  }
  sim.io.d.valid := VecInit(laneResps.map(_.valid)).asUInt
  sim.io.d.is_store := VecInit(laneResps.map(_.bits.is_store)).asUInt
  sim.io.d.size := VecInit(laneResps.map(_.bits.size)).asUInt
  sim.io.d.data := VecInit(laneResps.map(_.bits.data)).asUInt

  val sourceGens = Seq.fill(numLanes)(
    Module(
      new SourceGenerator(
        log2Ceil(numSrcIds),
        ignoreInUse = false
      )
    )
  )
  val anyInflight = sourceGens.map(_.io.inflight).reduce(_ || _)
  sim.io.inflight := anyInflight

  // Take requests off of the queue and generate TL requests
  (outer.laneNodes zip (laneReqs zip laneResps)).zipWithIndex.foreach {
    case ((node, (req, resp)), lane) =>
      val (tlOut, edge) = node.out(0)

      // Requests --------------------------------------------------------------
      //
      // Core only makes accesses of granularity larger than a word, so we want
      // the trace driver to act so as well.
      // That means if req.size is smaller than word size, we need to pad data
      // with zeros to generate a word-size request, and set mask accordingly.
      val offsetInWord = req.bits.address % wordSizeInBytes.U
      val subword = req.bits.size < log2Ceil(wordSizeInBytes).U

      // `mask` is currently unused
      // val mask = Wire(UInt(wordSizeInBytes.W))
      val wordData = Wire(UInt((wordSizeInBytes * 8 * 2).W))
      val sizeInBytes = Wire(UInt((sizeW + 1).W))
      sizeInBytes := (1.U) << req.bits.size
      // mask := Mux(subword, (~((~0.U(64.W)) << sizeInBytes)) << offsetInWord, ~0.U)
      wordData := Mux(subword, req.bits.data << (offsetInWord * 8.U), req.bits.data)
      val wordAlignedAddress =
        req.bits.address & ~((1 << log2Ceil(wordSizeInBytes)) - 1).U(addrW.W)
      val wordAlignedSize = Mux(subword, 2.U, req.bits.size)

      val sourceGen = sourceGens(lane)
      sourceGen.io.gen := tlOut.a.fire
      sourceGen.io.reclaim.valid := tlOut.d.fire
      sourceGen.io.reclaim.bits := tlOut.d.bits.source
      sourceGen.io.meta := DontCare

      val (plegal, pbits) = edge.Put(
        fromSource = sourceGen.io.id.bits,
        toAddress = wordAlignedAddress,
        lgSize = wordAlignedSize, // trace line already holds log2(size)
        // data should be aligned to beatBytes
        data =
          (wordData << (8.U * (wordAlignedAddress % edge.manager.beatBytes.U))).asUInt
      )
      val (glegal, gbits) = edge.Get(
        fromSource = sourceGen.io.id.bits,
        toAddress = wordAlignedAddress,
        lgSize = wordAlignedSize
      )
      val legal = Mux(req.bits.is_store, plegal, glegal)
      val bits = Mux(req.bits.is_store, pbits, gbits)

      tlOut.a.valid := req.valid && sourceGen.io.id.valid
      req.ready := tlOut.a.ready && sourceGen.io.id.valid

      when(tlOut.a.fire) {
        assert(legal, "illegal TL req gen")
      }
      tlOut.a.bits := bits

      // Responses -------------------------------------------------------------
      //
      tlOut.d.ready := resp.ready
      resp.valid := tlOut.d.valid
      resp.bits.is_store := !edge.hasData(tlOut.d.bits)
      resp.bits.size := tlOut.d.bits.size

      tlOut.b.ready := true.B
      tlOut.c.valid := false.B
      tlOut.e.valid := false.B

      // debug
      dontTouch(req)
      when(tlOut.a.valid) {
        printf(s"Lane ${lane}: ");
        TLPrintf(
          "Emulator",
          tlOut.a.bits.source,
          tlOut.a.bits.address,
          tlOut.a.bits.size,
          tlOut.a.bits.mask,
          req.bits.is_store,
          tlOut.a.bits.data,
          req.bits.data
        )
      }
      dontTouch(tlOut.a)
      dontTouch(tlOut.d)
  }

  // when(traceFinished && allReqReclaimed && noValidReqs) {
  //   assert(
  //     false.B,
  //     "\n\n\nsimulation Successfully finished\n\n\n (this assertion intentional fail upon MemTracer termination)"
  //   )
  // }
}

class SimEmulator(numLanes: Int)
    extends BlackBox(Map("NUM_LANES" -> numLanes))
    with HasBlackBoxResource {
  val traceLineT = new TraceLine
  val addrW = traceLineT.address.getWidth
  val sizeW = traceLineT.size.getWidth
  val dataW = traceLineT.data.getWidth
  val io = IO(new Bundle {
    val clock = Input(Clock())
    val reset = Input(Bool())
    val inflight = Input(Bool())
    val finished = Output(Bool())

    val a =
      new Bundle {
        val ready = Input(UInt(numLanes.W))
        val valid = Output(UInt(numLanes.W))
        // Chisel can't interface with Verilog 2D port, so flatten all lanes into
        // single wide 1D array.
        val address = Output(UInt((addrW * numLanes).W))
        val is_store = Output(UInt(numLanes.W))
        val size = Output(UInt((sizeW * numLanes).W))
        val data = Output(UInt((dataW * numLanes).W))
      }
    val d =
      new Bundle {
        val ready = Output(UInt(numLanes.W))
        val valid = Input(UInt(numLanes.W))
        val is_store = Input(UInt(numLanes.W))
        val size = Input(UInt((sizeW * numLanes).W))
        val data = Input(UInt((dataW * numLanes).W))
      }
  })

  addResource("/vsrc/SimDefaults.vh")
  addResource("/vsrc/SimEmulator.v")
  addResource("/csrc/SimEmulator.cc")
}

