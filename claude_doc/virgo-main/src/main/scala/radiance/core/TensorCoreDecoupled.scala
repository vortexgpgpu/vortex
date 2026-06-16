// See LICENSE.SiFive for license details.
// See LICENSE.Berkeley for license details.

package radiance.core

import chisel3._
import chisel3.util._
import chisel3.experimental.requireIsChiselType
import org.chipsalliance.cde.config.Parameters
import org.chipsalliance.diplomacy.lazymodule.{LazyModule, LazyModuleImp}
import freechips.rocketchip.tilelink._
import freechips.rocketchip.diplomacy.{IdRange, AddressSet}
import freechips.rocketchip.unittest.{UnitTest, UnitTestModule}
import radiance.memory.SourceGenerator

case class TensorTilingParams(
  // Dimension of the SMEM tile
  m: Int,
  n: Int,
  k: Int,
  // Dimension of the compute tile.  This is determined by the number of MAC
  // units
  mc: Int,
  nc: Int,
  kc: Int,
)

object TensorTilingParams {
  def fp16: TensorTilingParams = {
    TensorTilingParams (
      m = 16, n = 16, k = 32,
      mc = 4, nc = 4, kc = 8
    )
  }
  def fp32: TensorTilingParams = {
    TensorTilingParams (
      m = 16, n = 16, k = 16,
      mc = 4, nc = 4, kc = 4
    )
  }
}

class TensorCoreDecoupled(
    val numWarps: Int,
    val numLanes: Int,
    val half: Boolean, // input datatype is FP16 if true, FP32 if false
    val numSourceIds: Int = 16,
    val numFPRegs: Int = 32
) extends Module {
  val tilingParams =
    if (half) TensorTilingParams.fp16 else TensorTilingParams.fp32
  val numWarpBits = log2Ceil(numWarps)
  val wordSize = if (half) 2 else 4
  val wordSizeInBits = wordSize * 8/*bits*/
  val sourceWidth = log2Ceil(numSourceIds)
  val laneWidth = 4/*bytes*/ * 8/*bits*/
  val memWidth = numLanes * laneWidth
  val numFPRegBits = log2Ceil(numFPRegs)
  val addressWidth = 32

  val io = IO(new Bundle {
    val initiate = Flipped(Decoupled(new Bundle {
      val wid = UInt(numWarpBits.W)
      // SMEM start address of A and B tile
      val addressA = UInt(addressWidth.W)
      val addressB = UInt(addressWidth.W)
    }))
    val writeback = Decoupled(new Bundle {
      val last = Bool()
      val wid = UInt(numWarpBits.W)
      val rd = UInt(numFPRegBits.W)
      val data = Vec(numLanes, UInt(laneWidth.W))
    })
    val respA = Flipped(Decoupled(new TensorMemResp(sourceWidth, memWidth)))
    val respB = Flipped(Decoupled(new TensorMemResp(sourceWidth, memWidth)))
    val respC = Input(UInt(memWidth.W))
    val reqA = Decoupled(new TensorMemReq(sourceWidth))
    val reqB = Decoupled(new TensorMemReq(sourceWidth))
    val reqC = Output(Valid(UInt(numFPRegBits.W)))
  })
  dontTouch(io)

  class TensorMemReq(
    sourceWidth: Int
  ) extends Bundle {
    val source = UInt(sourceWidth.W)
    val address = UInt(addressWidth.W)
  }
  class TensorMemResp(
    sourceWidth: Int,
    dataWidth: Int
  ) extends Bundle {
    val source = UInt(sourceWidth.W)
    val data = UInt(dataWidth.W)
  }
  class TensorMemTag extends Bundle {
    val warp = UInt(numWarpBits.W)
    val set = UInt(setBits.W)
    val index = UInt(indexBits.W)
  }
  // mem response after translation from TL source to set/step tag
  class TensorMemRespWithTag(
    dataWidth: Int
  ) extends Bundle {
    val tag = new TensorMemTag
    val data = UInt(dataWidth.W)
  }

  // ===========================================================================
  // Access stage
  // ===========================================================================
  //
  // Frontend of the decoupled access/execute pipeline.

  // sets: k iteration
  val numSets = (tilingParams.k / tilingParams.kc)
  val setBits = log2Ceil(numSets)
  // steps: i-j iteration
  val numSteps = (tilingParams.m * tilingParams.n) / (tilingParams.mc * tilingParams.nc)
  val stepBits = log2Ceil(numSteps)
  val lastSet = ((1 << setBits) - 1)
  val lastStep = ((1 << stepBits) - 1)
  def setDone(set: UInt) = (set === lastSet.U)
  def stepDone(step: UInt) = (step === lastStep.U)
  // 'index' is the index of a memory request among the sequence of requests
  // needed to read a full M-column of A or N-row of B.  Its range is [0,m/2)
  // or [0,n/2), where 2 is the stride can be read in a single request size.
  require(tilingParams.m == tilingParams.n,
          "currently only supports square SMEM tile")
  val numIndices = tilingParams.m / 2/*FIXME:hardcoded?*/
  val indexBits = log2Ceil(numIndices)
  val lastIndex = (1 << indexBits) - 1

  object State extends ChiselEnum {
    val idle = Value(0.U)
    val run = Value(1.U)
    // All set/step sequencing is complete and the tensor core is holding the
    // result data until downstream writeback is ready.
    // FIXME: is this necessary if writeback is decoupled with queues?
    val finish = Value(2.U)
  }
  val state = RegInit(State.idle)
  val allReqsDone = WireInit(false.B)
  dontTouch(allReqsDone)

  val warpAccess = RegInit(0.U(numWarpBits.W))
  val addrAAccess = RegInit(0.U(addressWidth.W))
  val addrBAccess = RegInit(0.U(addressWidth.W))

  class BlockState extends Bundle {
    val set = UInt(setBits.W)
    val index = UInt(indexBits.W)
  }
  val stateInit = Wire(new BlockState)
  stateInit.set := 0.U
  stateInit.index := 0.U
  val stateA = RegInit(stateInit)
  val stateB = RegInit(stateInit)
  dontTouch(stateA)
  dontTouch(stateB)

  io.initiate.ready := (state === State.idle)
  when (io.initiate.fire) {
    warpAccess := io.initiate.bits.wid
    addrAAccess := io.initiate.bits.addressA
    addrBAccess := io.initiate.bits.addressB
    assert(stateA.set === 0.U && stateA.index === 0.U &&
           stateB.set === 0.U && stateB.index === 0.U,
           "stateA and stateB not initialized to zero")
  }

  switch(state) {
    is(State.idle) {
      when(io.initiate.fire) {
        state := State.run
      }
    }
    is(State.run) {
      when (allReqsDone) {
        state := State.finish
      }
    }
    is(State.finish) {
      // FIXME: is finish state needed?
      state := State.idle
    }
  }

  when (io.reqA.fire) {
    when (stateA.index === lastIndex.U) {
      stateA.set := stateA.set + 1.U
    }
    stateA.index := stateA.index + 1.U
  }
  when (io.reqB.fire) {
    when (stateB.index === lastIndex.U) {
      stateB.set := stateB.set + 1.U
    }
    stateB.index := stateB.index + 1.U
  }

  // Address generation
  //
  def addressGen(base: UInt, set: UInt, index: UInt): UInt = {
    // A is assumed to be block-wise M-major, and B block-wise N-major, to
    // facilitate bank conflict-free SMEM accesses.  With these layouts, the
    // same code below works for both A and B.
    //
    // a "block" is the 4*8 byte-sized contiguous memory that can be read in
    // one SMEM request.  The A and B matrix is assumed to be stored in
    // block-wise "index"-major order (M-major for A, N-major for B)
    val blockRow = set
    val blockCol = index
    val blockIndex = (blockRow << indexBits) + blockCol
    val blockSize = numLanes * (laneWidth / 8/*bits*/)
    val blockSizeBits = log2Ceil(blockSize)
    val byteOffset = blockIndex << blockSizeBits
    base + byteOffset

    // address generation for byte-wise K-major A and B layout
    // val elemRow = blockRow << 1
    // val elemCol =  blockCol << log2Ceil(tilingParams.kc)
    // val rowStride = tilingParams.k * wordSize
    // val rowStrideBits = log2Ceil(rowStride)
    // val wordStrideBits = log2Ceil(wordSize)
    // val tileOffset = (elemRow << rowStrideBits) + (elemCol << wordStrideBits)
    // base + tileOffset
  }

  val addressA = addressGen(addrAAccess, stateA.set, stateA.index)
  val addressB = addressGen(addrBAccess, stateB.set, stateB.index)

  val doneReqA = RegInit(false.B)
  val doneReqB = RegInit(false.B)
  val genReqA = (state === State.run) && !doneReqA
  val genReqB = (state === State.run) && !doneReqB

  // Request generation
  //
  val tagA = Wire(new TensorMemTag)
  tagA.warp := warpAccess
  tagA.set := stateA.set
  tagA.index := stateA.index
  val tagB = Wire(new TensorMemTag)
  tagB.warp := warpAccess
  tagB.set := stateB.set
  tagB.index := stateB.index

  val respATagged = Wire(Decoupled(new TensorMemRespWithTag(memWidth)))
  val respBTagged = Wire(Decoupled(new TensorMemRespWithTag(memWidth)))
  Seq((io.reqA, (io.respA, respATagged)),
      (io.reqB, (io.respB, respBTagged))).zipWithIndex.foreach {
    case ((req, (resp, respTagged)), i) => {
      val sourceGen = Module(new SourceGenerator(
        log2Ceil(numSourceIds),
        metadata = Some(new TensorMemTag)
      ))

      sourceGen.io.gen := req.fire
      sourceGen.io.meta := (if (i == 0) tagA else tagB)
      req.valid := (if (i == 0) genReqA else genReqB)
      req.bits.address := (if (i == 0) addressA else addressB)
      req.bits.source := sourceGen.io.id.bits

      sourceGen.io.reclaim.valid := resp.fire
      sourceGen.io.reclaim.bits := resp.bits.source

      // translate source
      respTagged.valid := resp.valid
      respTagged.bits.tag := sourceGen.io.peek
      respTagged.bits.data := resp.bits.data
      resp.ready := respTagged.ready
    }
  }

  // finalize state when everything has been accessed
  val lastReqA = (stateA.set === lastSet.U) && (stateA.index === lastIndex.U)
  val lastReqB = (stateB.set === lastSet.U) && (stateB.index === lastIndex.U)
  when (lastReqA && io.reqA.fire) { doneReqA := true.B }
  when (lastReqB && io.reqB.fire) { doneReqB := true.B }
  when (state === State.finish) {
    doneReqA := false.B
    doneReqB := false.B
    stateA.set := 0.U
    stateA.index := 0.U
    stateB.set := 0.U
    stateB.index := 0.U
  }

  allReqsDone := doneReqA && doneReqB

  // ===========================================================================
  // Execute stage
  // ===========================================================================
  //
  // Backend of the decoupled access/execute pipeline.
  //
  val respQueueDepth = 4 // FIXME: parameterize
  require(respQueueDepth >= 4,
    "respQueueDepth must be at least 4.  This is because the B operand buffer " ++
    "is shallower than A's, so the B response queue has to be deep enough to " ++
    "hold younger requests until A operand buffer becomes valid and the first DPU " ++
    "fire can happen.  FIXME: make operand buffer report per-subtile valid so " ++
    "the first compute can happen earlier.")
  val respQueueA = Queue(respATagged, respQueueDepth)
  val respQueueB = Queue(respBTagged, respQueueDepth)

  require(respQueueA.bits.data.widthOption.get ==
          io.writeback.bits.data.widthOption.get,
          "response data width does not match the writeback data width")

  // Stage the operands in a pipeline so that we obtain the full 4x4 tiles
  // ready for compute.  Also send the set/step tag along the pipe for
  // alignment check.

  // @cleanup: dedup A and B below

  val fullA = Module(new FillBuffer(
    chiselTypeOf(respQueueB.bits.data), numIndices
  ))
  fullA.io.enq.valid := respQueueA.valid
  fullA.io.enq.bits := respQueueA.bits.data
  respQueueA.ready := fullA.io.enq.ready
  // `pipe` combinationally couples enq-deq ready
  val fullATag = Module(new Queue(
    new TensorMemTag, entries = 1, pipe = true
  ))
  fullATag.io.enq.valid := respQueueA.valid
  fullATag.io.enq.bits := respQueueA.bits.tag

  // stage the full A tile once more so that FillBuffer can be filled up in the
  // background while the tile is being used for compute.  This does come with
  // capacity overhead.
  val fullABuf = Module(new Queue(
    new Bundle {
      val data = chiselTypeOf(fullA.io.deq.bits)
      val tag = new TensorMemTag
    }, entries = 1, pipe = true
  ))
  fullABuf.io.enq.valid := fullA.io.deq.valid
  fullABuf.io.enq.bits.data := fullA.io.deq.bits
  fullABuf.io.enq.bits.tag := fullATag.io.deq.bits
  fullA.io.deq.ready := fullABuf.io.enq.ready
  fullATag.io.deq.ready := fullABuf.io.enq.ready

  // serialize every two B responses into one full 4x4 B tile
  // FIXME: do the same for A
  val fullB = Module(new FillBuffer(
    chiselTypeOf(respQueueB.bits.data), 2/*substeps*/
  ))
  fullB.io.enq.valid := respQueueB.valid
  fullB.io.enq.bits := respQueueB.bits.data
  respQueueB.ready := fullB.io.enq.ready
  val fullBTag = Module(new Queue(
    new TensorMemTag, entries = 1, pipe = true
  ))
  fullBTag.io.enq.valid := respQueueB.valid
  fullBTag.io.enq.bits := respQueueB.bits.tag

  val fullBBuf = Module(new Queue(
    new Bundle {
      val data = chiselTypeOf(fullB.io.deq.bits)
      val tag = new TensorMemTag
    }, entries = 1, pipe = true
  ))
  fullBBuf.io.enq.valid := fullB.io.deq.valid
  fullBBuf.io.enq.bits.data := fullB.io.deq.bits
  fullBBuf.io.enq.bits.tag := fullBTag.io.deq.bits
  fullB.io.deq.ready := fullBBuf.io.enq.ready
  fullBTag.io.deq.ready := fullBBuf.io.enq.ready

  // C access from regfile
  //
  // C access is initiated from the backend because the regfile access latency
  // is much lower than the SMEM latency.  If C access is done in the frontend,
  // the C response queue will fill up and block further run-ahead until the A
  // and B smem reqs arrive and backend starts running, causing the frontend to
  // stall waiting for remaining C accesses to finish.

  // regfile is fast; don't need a deep response queue
  val respQueueCDepth = 3
  val respQueueC = Module(new Queue(
    new Bundle {
      val tag = new Bundle {
        val warp = UInt(numWarpBits.W)
        val set = UInt(setBits.W)
        val step = UInt(stepBits.W)
      }
      val data = UInt(io.respC.widthOption.get.W)
    },
    respQueueCDepth
  ))

  // access state of the C accumulator value
  val stateAccessC = RegInit(State.idle)
  // note this is different from warpAccess and belongs to execute
  val warpAccessC = RegInit(0.U(numWarpBits.W))
  val setAccessC = RegInit(0.U(setBits.W))
  val stepAccessC = RegInit(0.U(stepBits.W))
  val substepAccessC = RegInit(0.U(1.W))
  val genReqC = WireInit(false.B)
  val nextStepAccessC = genReqC && (substepAccessC === 1.U)
  when (genReqC) {
    substepAccessC := substepAccessC + 1.U
  }
  dontTouch(stepAccessC)
  dontTouch(substepAccessC)

  val doneReqC = RegInit(false.B)
  val lastReqC = (setAccessC === lastSet.U) && (stepAccessC === lastStep.U)
  when (lastReqC && nextStepAccessC) { doneReqC := true.B }

  switch(stateAccessC) {
    is(State.idle) {
      //arrival of A and B response kicks off the execute backend
      when (respQueueA.valid || respQueueB.valid) {
        stateAccessC := State.run
        when (respQueueA.valid) {
          warpAccessC := respQueueA.bits.tag.warp
        }.elsewhen (respQueueB.valid) {
          warpAccessC := respQueueB.bits.tag.warp
        }
      }
    }
    is(State.run) {
      when (doneReqC) {
        stateAccessC := State.finish
      }
    }
    is(State.finish) {
      // FIXME: is finish state needed?
      stateAccessC := State.idle
    }
  }

  when (stateAccessC === State.finish) {
    doneReqC := false.B
    setAccessC := 0.U
    stepAccessC := 0.U
    substepAccessC := 0.U
  }

  // because C reg IO arrives 1 cycle later, it will get latched onto the
  // response queue 2 cycles later.  Make sure there's at least two entries of
  // space at the queue before requesting the reg value
  // FIXME: doesn't feel very clean
  require(respQueueC.entries >= 2)
  val hasSpace =
    respQueueC.io.count <= Mux(respQueueC.io.deq.fire,
                               (respQueueC.entries - 1).U,
                               (respQueueC.entries - 2).U)

  genReqC := (stateAccessC === State.run) && hasSpace && !doneReqC

  // give 1-cycle delay to sync valid/metadata with C regfile response
  val respCValid = RegNext(genReqC)
  val warpAccessCDelayed = RegNext(warpAccessC)
  val setAccessCDelayed = RegNext(setAccessC)
  val stepAccessCDelayed = RegNext(stepAccessC)

  // note rd is independent to sets
  def rdGen(step: UInt, substep: UInt): UInt = {
    // each step produces 4x4 output tile, written by 8 threads with 2 regs per
    // thread
    (step << 1/*2 substeps*/) + substep
  }
  io.reqC.valid := genReqC
  io.reqC.bits := rdGen(stepAccessC, substepAccessC)

  // queue the regfile response to buffers
  // these strictly belong to the execute stage
  respQueueC.io.enq.valid := respCValid
  respQueueC.io.enq.bits.tag.warp := warpAccessCDelayed
  respQueueC.io.enq.bits.tag.set := setAccessCDelayed
  respQueueC.io.enq.bits.tag.step := stepAccessCDelayed
  respQueueC.io.enq.bits.data := io.respC

  // serialize every two C responses into one full 4x4 C tile
  val fullC = Module(new FillBuffer(
    chiselTypeOf(respQueueC.io.deq.bits.data), 2/*substeps*/
  ))
  fullC.io.enq.valid := respQueueC.io.deq.valid
  fullC.io.enq.bits := respQueueC.io.deq.bits.data
  respQueueC.io.deq.ready := fullC.io.enq.ready
  val fullCTag = Module(new Queue(
    chiselTypeOf(respQueueC.io.deq.bits.tag),
    entries = 1, pipe = true
  ))
  fullCTag.io.enq.valid := respQueueC.io.deq.valid
  fullCTag.io.enq.bits := respQueueC.io.deq.bits.tag

  val fullCBuf = Module(new Queue(
    new Bundle {
      val tag = chiselTypeOf(fullCTag.io.deq.bits)
      val data = chiselTypeOf(fullC.io.deq.bits)
    }, entries = 1, pipe = true
  ))
  fullCBuf.io.enq.valid := fullC.io.deq.valid
  fullCBuf.io.enq.bits.data := fullC.io.deq.bits
  fullCBuf.io.enq.bits.tag := fullCTag.io.deq.bits
  fullC.io.deq.ready := fullCBuf.io.enq.ready
  fullCTag.io.deq.ready := fullCBuf.io.enq.ready

  val dpuReady = Wire(Bool())
  val dpuFire = Wire(Bool())
  val operandsValid =
    fullABuf.io.deq.valid && fullBBuf.io.deq.valid && fullCBuf.io.deq.valid
  dpuFire := operandsValid && dpuReady
  dontTouch(dpuFire)

  val setCompute = RegInit(0.U(setBits.W))
  val stepCompute = RegInit(0.U(stepBits.W))
  val substepCompute = RegInit(0.U(1.W))
  val nextStepCompute = dpuFire && (substepCompute === 1.U)
  dontTouch(setCompute)
  dontTouch(stepCompute)
  dontTouch(substepCompute)
  when (dpuFire) {
    substepCompute := substepCompute + 1.U
  }

  // Operand selection
  //
  // select the correct 4x4 tile from A operand buffer
  val numTilesM = tilingParams.m / tilingParams.mc
  val numTilesMBits = log2Ceil(numTilesM)
  def selectOperandA(buf: Vec[UInt]): UInt = {
    require(buf.length == numIndices)
    val stepM = stepCompute & ((1 << numTilesMBits) - 1).U
    Cat(buf((stepM << 1) + 1.U), buf(stepM << 1))
  }
  val operandA = selectOperandA(fullABuf.io.deq.bits.data)
  val operandATag = fullABuf.io.deq.bits.tag
  // select the correct 2x4 tile from B/C operand buffer
  val operandB = fullBBuf.io.deq.bits.data(substepCompute)
  val operandBTag = fullBBuf.io.deq.bits.tag
  val operandC = fullCBuf.io.deq.bits.data(substepCompute)
  val operandCTag = fullCBuf.io.deq.bits.tag
  dontTouch(operandATag)
  dontTouch(operandBTag)
  dontTouch(operandCTag)

  // Operand buffer logic
  //
  // hold A data until the entire set is done
  val shouldDequeueAMask = ((1 << stepBits) - 1).U
  val shouldDequeueA =
    ((stepCompute & shouldDequeueAMask) === shouldDequeueAMask) &&
    (substepCompute === 1.U)
  fullABuf.io.deq.ready := dpuFire && shouldDequeueA
  // hold B tile at respQueueB for multiple steps for reuse, only dequeue when
  // we fully iterated a column (M-dimension)
  val shouldDequeueBMask = ((1 << numTilesMBits) - 1).U
  val shouldDequeueB =
    ((stepCompute & shouldDequeueBMask) === shouldDequeueBMask) &&
    (substepCompute === 1.U)
  fullBBuf.io.deq.ready := dpuFire && shouldDequeueB
  // C should be dequeued everytime a 4x4 tile is done computing
  val shouldDequeueC = (substepCompute === 1.U)
  fullCBuf.io.deq.ready := dpuFire && shouldDequeueC

  dontTouch(respQueueA)
  dontTouch(respQueueB)
  dontTouch(shouldDequeueA)
  dontTouch(shouldDequeueB)

  // Assert that the DPU is computing with operands of the same set/step. Note
  // that the B resp will only have step values multiple of 4 due to reuse.
  //
  // This check assumes that memory responses come back in-order.  Might be too
  // strong of an assumption depending on the backing memory.
  def assertAligned = {
    val stepMask = (1 << numTilesMBits).U
    when (dpuFire) {
      assert(operandATag.warp === operandBTag.warp &&
             operandATag.set  === operandBTag.set,
             "A and B operands are pointing to different warps and sets. " ++
             "This might indicate memory response coming back out-of-order.")
      assert(operandATag.warp === operandCTag.warp &&
             operandATag.set  === operandCTag.set,
             "A and C operands are pointing to different warps and sets. " ++
             "This might indicate memory response coming back out-of-order.")
      assert(operandATag.set === setCompute,
             "Operand arrived from memory is pointing at a different set than the FSM.")
    }
  }
  assertAligned

  // Dot-product unit
  //
  // 4x2 four-element DPUs summing up to 32 FP32 MACs in total
  //
  val ncSubstep = tilingParams.nc / 2
  require(tilingParams.mc * ncSubstep == numLanes,
          "substep tile size doesn't match writeback throughput")
  val dpus = Seq.fill(tilingParams.mc)(Seq.fill(ncSubstep)(
    Module(new TensorDotProductUnit(
      dim = tilingParams.kc,
      half = half,
    ))
  ))

  // reshape UInt into a two-dimensional array where the innermost dimension
  // has `numWords` elements
  def reshapeByWords(x: UInt, wordSizeInBits: Int, numWords: Int)
  : Seq[Seq[UInt]] = {
    x.asBools.grouped(wordSizeInBits).map(VecInit(_).asUInt).toSeq
     .grouped(numWords).toSeq
  }
  val operandADimensional =
    reshapeByWords(operandA, wordSizeInBits, tilingParams.kc)
  require(operandADimensional.length == tilingParams.mc &&
          operandADimensional(0).length == tilingParams.kc,
          "operand width doesn't agree with tiling parameter")
  val operandBDimensional =
    reshapeByWords(operandB, wordSizeInBits, tilingParams.kc)
  require(operandBDimensional.length == ncSubstep &&
          operandBDimensional(0).length == tilingParams.kc,
          "operand width doesn't agree with tiling parameter")
  // note operand C is M-major, and always FP32
  val operandCDimensional =
    reshapeByWords(operandC, 4/*fp32*/ * 8/*bits*/, tilingParams.mc)
  require(operandCDimensional.length == ncSubstep &&
          operandCDimensional(0).length == tilingParams.mc,
          "operand width doesn't agree with tiling parameter")

  for (m <- 0 until tilingParams.mc) {
    for (n <- 0 until ncSubstep) {
      dpus(m)(n).io.in.valid := dpuFire
      dpus(m)(n).io.in.bits.a := operandADimensional(m)
      dpus(m)(n).io.in.bits.b := operandBDimensional(n)
      dpus(m)(n).io.in.bits.c := operandCDimensional(n)(m)
      // dpu ready couples with writeback backpressure
      dpus(m)(n).io.stall := !io.writeback.ready
    }
  }
  dpuReady := !dpus(0)(0).io.stall
  dontTouch(dpuFire)
  dontTouch(dpuReady)

  val dpuValids = dpus.flatMap(_.map(_.io.out.valid))
  val dpuValid = dpuValids.reduce(_ && _)
  def assertDPU = {
    val dpuStalls = dpus.flatMap(_.map(_.io.stall))
    assert(dpuStalls.reduce(_ && _) === dpuStalls.reduce(_ || _),
      "stall signals of DPUs went out of sync")
    assert(dpuValids.reduce(_ && _) === dpuValids.reduce(_ || _),
      "valid signals of DPUs went out of sync")
  }
  assertDPU

  // flatten DPU output into 1D array in M-major order
  val flattenedDPUOut = (0 until ncSubstep).flatMap { n =>
    (0 until tilingParams.mc).map { m =>
      dpus(m)(n).io.out.bits.data
    }
  }
  io.writeback.bits.data := flattenedDPUOut

  // Writeback logic
  //
  // These queues hold metadata needed for writeback in sync with the DPU.

  class TensorComputeTag extends Bundle {
    val warp = UInt(numWarpBits.W)
    val set = UInt(setBits.W)
    val step = UInt(stepBits.W)
    val substep = UInt(1.W)
  }

  val queueDepth = (if (half) 6 else 5) // needs to be at least the DPU latency
  val tagQueue = Module(new Queue(new TensorComputeTag, queueDepth))
  tagQueue.io.enq.valid := dpuFire
  tagQueue.io.enq.bits.warp := operandATag.warp
  tagQueue.io.enq.bits.set := setCompute
  tagQueue.io.enq.bits.step := stepCompute
  tagQueue.io.enq.bits.substep := substepCompute
  tagQueue.io.deq.ready := io.writeback.fire
  // this is not necessary for correctness, and might trigger when there's a
  // lot of writeback contention
  assert(tagQueue.io.enq.ready === true.B,
         "tag queue full, DPU operation might be throttled")
  assert(!dpuValid || tagQueue.io.deq.valid,
         "tag queue and DPU went out of sync")

  // val widQueue = Queue(io.initiate, queueDepth, pipe = (queueDepth == 1))

  val warpWriteback = tagQueue.io.deq.bits.warp
  val setWriteback = tagQueue.io.deq.bits.set
  val stepWriteback = tagQueue.io.deq.bits.step
  val substepWriteback = tagQueue.io.deq.bits.substep
  io.writeback.valid := dpuValid
  io.writeback.bits.wid := warpWriteback
  io.writeback.bits.rd := rdGen(stepWriteback, substepWriteback)
  io.writeback.bits.last := setDone(setWriteback) && stepDone(stepWriteback) &&
                            (substepWriteback === 1.U)

  // State transition
  // ----------------
  //
  // set/step sequencing logic

  def sequenceSetStep(set: UInt, step: UInt, nextStep: Bool) = {
    when (nextStep) {
      step := (step + 1.U) & lastStep.U
      when (stepDone(step)) {
        set := (set + 1.U) & lastSet.U
      }
    }
  }
  sequenceSetStep(setAccessC, stepAccessC, nextStepAccessC)
  sequenceSetStep(setCompute, stepCompute, nextStepCompute)
}

// A buffer that collects multiple entries of input data and exposes the
// coalesced data as output.  Effectively acts as a width-widening
// chisel.util.Pipe.
class FillBuffer[T <: Data](
  val gen: T,
  val entries: Int,
  val pipe: Boolean = true,
) extends Module {
  require(entries > 1, "FillBuffer must have more than one entries")
  requireIsChiselType(gen)

  val io = IO(new Bundle {
    val enq = Flipped(Decoupled(gen))
    val deq = Decoupled(Vec(entries, gen))
  })

  val data = Reg(Vec(entries, gen))
  val ptr = Counter(entries)
  dontTouch(ptr.value)
  // val full = (ptr.value === entries.U)
  val full = RegInit(false.B)
  io.enq.ready := !full
  if (pipe) {
    when(io.deq.fire) { io.enq.ready := true.B }
  }

  when (io.enq.fire) {
    when (ptr.value === (entries - 1).U) {
      full := true.B
    }
    // if buffer was full and pipe was true, ptr.value would point to 0, and
    // enq will correctly write to the first element
    data(ptr.value) := io.enq.bits
    ptr.inc()
  }
  io.deq.valid := full
  when (io.deq.fire) {
    assert(full === true.B, "FillBuffer fired before buffer was full")
    assert(ptr.value === 0.U, "ptr did not wrap around when FillBuffer fired")
    // TODO: if entries == 1, full might stay high if enq also fired
    full := false.B
  }
  (io.deq.bits zip data).foreach { case (io, d) => io := d }
}

// synthesizable unit tests

// wraps TensorCoreDecoupled with a TileLink client node for use in a Diplomacy
// graph.
class TensorCoreDecoupledTL(implicit p: Parameters) extends LazyModule {
  val numSourceIds = 16

  // node with two edges; one for A and one for B matrix
  val node = TLClientNode(Seq(
    TLMasterPortParameters.v2(
      Seq(TLMasterParameters.v2(
        name = "TensorCoreDecoupledMatrixANode",
        sourceId = IdRange(0, numSourceIds)
      ))
    ),
    TLMasterPortParameters.v2(
      Seq(TLMasterParameters.v2(
        name = "TensorCoreDecoupledMatrixBNode",
        sourceId = IdRange(0, numSourceIds)
      ))
    )
  ))

  lazy val module = new TensorCoreDecoupledTLImp(this)
}

class TensorCoreDecoupledTLImp(outer: TensorCoreDecoupledTL)
    extends LazyModuleImp(outer) with UnitTestModule {
  require(outer.node.out.length == 2/*A and B*/)

  val tensor = Module(new TensorCoreDecoupled(
                      8, 8, half = true, outer.numSourceIds))
  val wordSize = 4 // @cleanup: hardcoded

  val zip = Seq((outer.node.out(0), tensor.io.reqA),
                (outer.node.out(1), tensor.io.reqB))
  zip.foreach { case ((tl, edge), req) =>
    tl.a.valid := req.valid
    val (legal, bits) = edge.Get(
      fromSource = req.bits.source,
      toAddress = req.bits.address,
      lgSize = log2Ceil(wordSize).U
    )
    tl.a.bits := bits
    req.ready := tl.a.ready
    when(tl.a.fire) {
      assert(legal, "illegal TL req gen")
    }
  }

  // TODO: dedup A and B
  val (tlOutA, _) = outer.node.out(0)
  val (tlOutB, _) = outer.node.out(1)
  tensor.io.respA.valid := tlOutA.d.valid
  tensor.io.respA.bits.data := tlOutA.d.bits.data
  tensor.io.respA.bits.source := tlOutA.d.bits.source
  tlOutA.d.ready := tensor.io.respA.ready
  tensor.io.respB.valid := tlOutB.d.valid
  tensor.io.respB.bits.data := tlOutB.d.bits.data
  tensor.io.respB.bits.source := tlOutB.d.bits.source
  tlOutB.d.ready := tensor.io.respB.ready

  tensor.io.respC := 42.U // FIXME bogus

  tensor.io.initiate.valid := io.start
  tensor.io.initiate.bits.wid := 3.U // bogus, static value
  tensor.io.initiate.bits.addressA := 0x0.U
  tensor.io.initiate.bits.addressB := 0x800.U
  tensor.io.writeback.ready := true.B

  io.finished := tensor.io.writeback.valid && tensor.io.writeback.bits.last
  when (io.finished) {
    // might be too strong
    assert(tensor.io.writeback.bits.rd === 31.U)
  }
}

// a minimal Diplomacy graph with a tensor core and a TLRAM
class TensorCoreDecoupledTLRAM(implicit p: Parameters) extends LazyModule {
  val tensor = LazyModule(new TensorCoreDecoupledTL)
  val xbar = LazyModule(new TLXbar)
  val ram = LazyModule(new TLRAM(
    address = AddressSet(0x0000, 0xffffff),
    beatBytes = 32 // @cleanup: hardcoded
  ))

  ram.node :=* xbar.node :=* tensor.node

  lazy val module = new Impl
  class Impl extends LazyModuleImp(this) with UnitTestModule {
    tensor.module.io.start := io.start
    io.finished := tensor.module.io.finished
  }
}

// two separate TLRAMs for A and B for full throughput
class TensorCoreDecoupledTwoTLRAM(implicit p: Parameters) extends LazyModule {
  val tensor = LazyModule(new TensorCoreDecoupledTL)
  val xbar = LazyModule(new TLXbar)
  val ramA = LazyModule(new TLRAM(
    address = AddressSet(0x000, 0xfffbff),
    beatBytes = 32 // @cleanup: hardcoded
  ))
  val ramB = LazyModule(new TLRAM(
    address = AddressSet(0x400, 0xfffbff),
    beatBytes = 32 // @cleanup: hardcoded
  ))

  val stutter = new TLIdentityNode
  xbar.node :=* tensor.node
  ramA.node := stutter := xbar.node
  ramB.node := xbar.node

  val fuzz = false

  lazy val module = new Impl
  class Impl extends LazyModuleImp(this) with UnitTestModule {
    tensor.module.io.start := io.start
    io.finished := tensor.module.io.finished

    val (tlIn, _) = stutter.in(0)
    val (tlOut, _) = stutter.out(0)
    require(stutter.in.length == 1)
    require(stutter.out.length == 1)

    // inject stalls for fuzzing
    val incr = Wire(Bool())
    val (count, _) = Counter(incr, 0x1000)
    def cond(x: UInt) = (x & ((1 << 3) - 1).U) =/= 0.U
    val stall = if (fuzz) cond(count) else false.B

    tlOut.a <> tlIn.a
    tlIn.d <> tlOut.d
    incr := tlIn.a.fire || stall
    when (stall) {
      tlIn.a.ready := false.B
      tlOut.a.valid := false.B
    }
  }
}

// unit test harness
class TensorCoreDecoupledTest(timeout: Int = 500000)(implicit p: Parameters)
    extends UnitTest(timeout) {
  // val dut = Module(LazyModule(new TensorCoreDecoupledTLRAM).module)
  val dut = Module(LazyModule(new TensorCoreDecoupledTwoTLRAM).module)
  dut.io.start := io.start
  io.finished := dut.io.finished
}
