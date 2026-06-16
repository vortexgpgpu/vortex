// See LICENSE.SiFive for license details.
// See LICENSE.Berkeley for license details.

package radiance.tile

import chisel3._
import chisel3.util._
import chisel3.experimental.BundleLiterals._
import org.chipsalliance.diplomacy.DisableMonitors
import org.chipsalliance.diplomacy.lazymodule._
import freechips.rocketchip.diplomacy.{AddressSet, SimpleDevice}
import freechips.rocketchip.resources.BigIntHexContext
import freechips.rocketchip.prci.{ClockCrossingType, ClockSinkParameters}
import freechips.rocketchip.regmapper.RegField
import freechips.rocketchip.rocket._
import freechips.rocketchip.subsystem.{CanAttachTile, HierarchicalElementCrossingParamsLike, RocketCrossingParams}
import freechips.rocketchip.tile._
import freechips.rocketchip.tilelink._
import gemmini._
import org.chipsalliance.cde.config.Parameters
import radiance.subsystem.{GPUMemParams, GPUMemory}

case class GemminiCoreParams(
  useVM: Boolean = false,
  useHypervisor: Boolean = false,
  useUser: Boolean = false,
  useSupervisor: Boolean = false,
  useDebug: Boolean = false,
  useAtomics: Boolean = false,
  useAtomicsOnlyForIO: Boolean = false,
  useCompressed: Boolean = false,
  useRVE: Boolean = false,
  mulDiv: Option[MulDivParams] = None,
  fpu: Option[FPUParams] = None,
  fetchWidth: Int = 1,
  decodeWidth: Int = 1,
  retireWidth: Int = 1,
  instBits: Int = 0,
  nLocalInterrupts: Int = 0,
  nPMPs: Int = 0,
  nBreakpoints: Int = 0,
  useBPWatch: Boolean = false,
  nPerfCounters: Int = 0,
  haveBasicCounters: Boolean = false,
  haveFSDirty: Boolean = false,
  misaWritable: Boolean = false,
  haveCFlush: Boolean = false,
  nL2TLBEntries: Int = 0,
  mtvecInit: Option[BigInt] = Some(BigInt(0)),
  mtvecWritable: Boolean = false,
  nL2TLBWays: Int = 0,
  lrscCycles: Int = 8,
  mcontextWidth: Int = 0,
  scontextWidth: Int = 0,
  useNMI: Boolean = false,
  nPTECacheEntries: Int = 0,
  traceHasWdata: Boolean = false,
  useConditionalZero: Boolean = false,
  bootFreqHz: BigInt = 0,
  pmpGranularity: Int = 0) extends CoreParams {
}

case class GemminiTileParams(
    tileId: Int = 0,
    gemminiConfig: GemminiArrayConfig[Float, Float, Float],
    tileSize: Either[(Int, Int, Int), Int] = Right(4),
    slaveAddress: BigInt
) extends InstantiableTileParams[GemminiTile] {
  def instantiate(crossing: HierarchicalElementCrossingParamsLike, lookup: LookupByHartIdImpl)(
      implicit p: Parameters
  ): GemminiTile = {
    new GemminiTile(this, crossing, lookup)
  }
  val core = GemminiCoreParams()
  val name = Some("radiance_gemmini_tile")
  val clockSinkParams = ClockSinkParameters()
  val blockerCtrlAddr = None
  val icache = None
  val dcache = Some(DCacheParams(
    nSets = 1, nWays = 1, nMSHRs = 0,
    nTLBSets = 0, nTLBWays = 1
  ))
  val btb = None
  val baseName = name.get
  val uniqueName = s"${baseName}_$tileId"
}

case class GemminiTileAttachParams(
  tileParams: GemminiTileParams,
  crossingParams: RocketCrossingParams,
) extends CanAttachTile { type TileType = GemminiTile }

class GemminiTile private (
    val gemminiParams: GemminiTileParams,
    crossing: ClockCrossingType,
    lookup: LookupByHartIdImpl,
    q: Parameters
) extends BaseTile(gemminiParams, crossing, lookup, q)
    with SinksExternalInterrupts
    with SourcesExternalNotifications {

  def this(params: GemminiTileParams, crossing: HierarchicalElementCrossingParamsLike,
      lookup: LookupByHartIdImpl)(implicit p: Parameters) =
    this(params, crossing.crossingType, lookup, p)

  val cpuDevice: SimpleDevice = new SimpleDevice(s"gemmini${tileId}", Nil)

  val intOutwardNode = None
  val slaveNode = TLIdentityNode()
  val masterNode = visibilityNode
  // val statusNode = BundleBridgeSource(() => new GroundTestStatus)

  val accSlaveNode = AccSlaveNode()

  tlOtherMastersNode := tlMasterXbar.node
  masterNode :=* tlOtherMastersNode
  DisableMonitors { implicit p => tlSlaveXbar.node :*= slaveNode }

  // TODO: evaluate if gemmini write node is required at all
  val gemmini = LazyModule(new Gemmini(gemminiParams.gemminiConfig))
  val base = p(GPUMemory()) match {
    case Some(GPUMemParams(baseAddr, _)) => baseAddr
    case _ => BigInt(0)
  }
  // tlMasterXbar.node :=* AddressOrNode(base) :=* gemmini.atlNode
  // tlOtherMastersNode :=* AddressOrNode(base) :=* gemmini.tlNode
  tlMasterXbar.node :=* gemmini.atlNode
  tlOtherMastersNode :=* gemmini.tlNode
  // gemmini.stlNode := tlSlaveXbar.node

  require(!gemmini.config.sp_singleported, "external scratchpad must be dual ported")

  val regDevice = new SimpleDevice("gemmini-cmd-reg", Seq(s"gemmini-cmd-reg"))
  val regNode = TLRegisterNode(
    address = Seq(AddressSet(gemminiParams.slaveAddress, 0xff)),
    device = regDevice,
    beatBytes = 8,
    concurrency = 1)
  regNode := TLFragmenter(8, 64) := tlSlaveXbar.node

  override lazy val module = new GemminiTileModuleImp(this)
}

class GemminiTileModuleImp(outer: GemminiTile) extends BaseTileModuleImp(outer) {

  def tieOffGemminiRocc: Unit = {
    val gemmini_io = outer.gemmini.module.io
    gemmini_io.ptw <> DontCare
    gemmini_io.mem <> DontCare
    gemmini_io.resp <> DontCare
    gemmini_io.fpu_req.ready := false.B
    gemmini_io.fpu_resp.valid := false.B
    gemmini_io.fpu_resp.bits := DontCare
    gemmini_io.exception := DontCare
  }

  tieOffGemminiRocc

  val accSlave = outer.accSlaveNode.in.head._1

  val instCounter = Counter(4)
  val ciscValid = RegInit(false.B)
  val ciscArgs = RegInit(0.U(24.W))
  val ciscId = RegInit(0.U(8.W))
  val ciscInstT = new Bundle {
    val inst = UInt(32.W)
    val rs1 = UInt(64.W)
    val rs2 = UInt(64.W)
  }
  val ciscInst = Wire(ciscInstT)
  val startsLoop = WireInit(false.B)
  val runningLoops = RegInit(0.U(4.W))

  val accCommandQueue = Module(new Queue(UInt(32.W), 4, false, true))
  accCommandQueue.io.enq.bits := accSlave.cmd.bits
  accCommandQueue.io.enq.valid := accSlave.cmd.valid
  accCommandQueue.io.deq.ready := !ciscValid
  assert(!accSlave.cmd.valid || accCommandQueue.io.enq.ready, "cisc command queue full")

  when (accCommandQueue.io.enq.fire) {
    val enqId = accSlave.cmd.bits(6, 0)
    startsLoop := VecInit(Seq(0, 1, 2, 9, 10, 12).map { x => enqId === x.U }).asUInt.orR
  }

  when (accCommandQueue.io.deq.fire) {
    ciscValid := true.B
    ciscId := accCommandQueue.io.deq.bits(7, 0)
    ciscArgs := accCommandQueue.io.deq.bits(31, 8)
    instCounter.reset()
  }

  def microcodeEntry[T <: Data](insts: Seq[T]): T = {
    when (instCounter.value === (insts.size - 1).U) {
      ciscValid := false.B
      instCounter.reset()
    }.otherwise {
      instCounter.inc()
    }
    VecInit(insts)(instCounter.value)
  }

  ciscInst := 0.U.asTypeOf(ciscInstT)

  val (tileSizeM, tileSizeN, tileSizeK) = outer.gemminiParams.tileSize match {
    case Left(v: (Int, Int, Int)) => v
    case Right(v: Int) => (v, v, v)
  }
  val config = outer.gemminiParams.gemminiConfig
  val spadHexadecile = config.sp_bank_entries * config.sp_banks / 16

  // TODO: as a temporary hack, bit 7 of the cisc opcode
  // TODO: will force the tile size to be a square base on M.

  val rectBoundsInst = ciscInstT.Lit(_.inst -> 0x1220b07b.U, _.rs1 -> 0.U,
      _.rs2 -> (tileSizeM | (tileSizeN << 16) | (BigInt(tileSizeK) << 32)).U)
  val squareBoundsInst = ciscInstT.Lit(_.inst -> 0x1220b07b.U, _.rs1 -> 0.U,
      _.rs2 -> (tileSizeM | (tileSizeM << 16) | (BigInt(tileSizeM) << 32)).U)
  val boundsInst = Mux(ciscId(7), squareBoundsInst, rectBoundsInst)

  def genStrideInst(tileA: UInt, tileB: UInt) = {
    val inst = Wire(ciscInstT)
    inst.inst := 0x3020b07b.U
    inst.rs1 := tileA * spadHexadecile.U          // A should be stored from the start of this block
    inst.rs2 := (tileB + 1.U) * spadHexadecile.U  // B should be stored up till the end of this block
    inst
  }

  def genAccSkipInst(accumulate: UInt, skips: UInt) = {
    val inst = Wire(ciscInstT)
    inst.inst := 0x1020b07b.U
    inst.rs1 := accumulate
    inst.rs2 := skips
    inst
  }

  println(s"gemmini cisc initialized with DIM=${config.DIM}, tileSize=${tileSizeM},${tileSizeN},${tileSizeK}")
  println(f"boundsInst=${rectBoundsInst.litValue}%x, hexadecile=${spadHexadecile}")

  when (ciscValid) {
    switch (ciscId(6, 0)) {
      is (0.U) { // compute on given hexadeciles
        val strideInst = genStrideInst(ciscArgs(7, 0), ciscArgs(15, 8))
        val accSkipInst = genAccSkipInst(ciscArgs(16), 0x2b8.U)
        ciscInst := microcodeEntry(Seq(boundsInst, strideInst, accSkipInst))
      } // replaces opcode 0: (a, b, accum) = (0, 2, 0), op 1 = (0, 2, 1), op 2 = (1, 3, 1), op 3 = (1, 3, 0)
      is (1.U) { // compute on given hexadeciles and mvout to spad
        val strideInst = genStrideInst(ciscArgs(7, 0), ciscArgs(15, 8))
        // note that accumulation is disabled
        val accSkipInst = genAccSkipInst(0.U, ((ciscArgs(23, 16) * spadHexadecile.U) << 32).asUInt | 0x238.U)
        ciscInst := microcodeEntry(Seq(boundsInst, strideInst, accSkipInst))
      }
      is (2.U) {} // no actual invocation, fake job placeholder
      is (8.U) { // set a, b stride
        val inst = Wire(ciscInstT)
        inst.inst := 0x1820b07b.U
        inst.rs1 := ciscArgs(11, 0)  // a
        inst.rs2 := ciscArgs(23, 12) // b
        ciscInst := microcodeEntry(Seq(inst))
      }
      is (9.U) { // move out to scratchpad
        val accSkipInst = genAccSkipInst(0.U, ((ciscArgs(7, 0) * spadHexadecile.U) << 32).asUInt | 0x278.U)
        ciscInst := microcodeEntry(Seq(boundsInst, accSkipInst))
      }
      is (10.U) { // load to scratchpad hexadeciles
        val strideInst = genStrideInst(ciscArgs(7, 0), ciscArgs(15, 8))
        val accSkipInst = genAccSkipInst(1.U, 0x2e0.U)
        ciscInst := microcodeEntry(Seq(boundsInst, strideInst, accSkipInst))
      } // replaces opcode 10: (a, b) = (0, 2), opcode 11 = (1, 3), opcode 12 = (0, 0), opcode 13 = (2, 2)
      is (11.U) { // set d, c stride
        val inst = Wire(ciscInstT)
        inst.inst := 0x1a20b07b.U
        inst.rs1 := ciscArgs(11, 0)  // d
        inst.rs2 := ciscArgs(23, 12) // c
        ciscInst := microcodeEntry(Seq(inst))
      }
      is (12.U) { // store to gmem
        val accSkipInst = genAccSkipInst(0.U, 0x78.U)
        ciscInst := microcodeEntry(Seq(boundsInst, accSkipInst))
      }

      is (16.U) { // unused, configure gemmini
        ciscInst := microcodeEntry(Seq(
          ciscInstT.Lit(_.inst -> 0x0020b07b.U, _.rs1 -> x"3f800000_00080101".U, _.rs2 -> 0.U),
          ciscInstT.Lit(_.inst -> 0x0020b07b.U, _.rs1 -> x"3f800000_00010004".U, _.rs2 -> x"10000_00000000".U),
          ciscInstT.Lit(_.inst -> 0x0020b07b.U, _.rs1 -> 0x2.U, _.rs2 -> x"3f800000_00000000".U)
        ))
      }
    }
  }

  val completionCount = PopCount(outer.gemmini.module.completion_io.completed)
  val loopStarted = Mux(startsLoop, 1.U, 0.U)
  runningLoops := runningLoops + loopStarted - completionCount
  assert(runningLoops + loopStarted >= completionCount)

  val gemminiIO = outer.gemmini.module.io.cmd

  val regValid = Wire(Bool())
  val regCommand = Wire(gemminiIO.bits.inst.cloneType)
  val gemminiRs1RegLSB = RegInit(0.U(32.W))
  val gemminiRs1RegMSB = RegInit(0.U(32.W))
  val gemminiRs2RegLSB = RegInit(0.U(32.W))
  val gemminiRs2RegMSB = RegInit(0.U(32.W))

  def gemminiCommandReg(valid: Bool, bits: UInt): Bool = {
    regValid := valid
    regCommand := bits.asTypeOf(regCommand)
    gemminiIO.ready && !ciscValid
  }

  def gemminiBusyReg(_dReady: Bool): (Bool, UInt) = {
    // (aReady, bits)
    // (!outer.gemmini.module.io.busy, outer.gemmini.module.io.busy.asUInt)
    (true.B, outer.gemmini.module.io.busy.asUInt)
  }

  def gemminiRunningLoopsReg(_dReady: Bool): (Bool, UInt) = {
    (true.B, runningLoops)
  }

  outer.regNode.regmap(
    0x00 -> Seq(RegField.w(32, gemminiCommandReg(_, _))),
    0x10 -> Seq(
      RegField.w(32, gemminiRs1RegLSB),
      RegField.w(32, gemminiRs1RegMSB)),
    0x18 -> Seq(
      RegField.w(32, gemminiRs2RegLSB),
      RegField.w(32, gemminiRs2RegMSB)),
    0x20 -> Seq(RegField.r(32, gemminiBusyReg(_))),
    0x28 -> Seq(RegField.r(32, gemminiRunningLoopsReg(_)))
  )

  assert(!regValid || gemminiIO.ready)
  assert(!ciscValid || gemminiIO.ready)

  gemminiIO.bits.status := 0.U.asTypeOf(gemminiIO.bits.status)
  gemminiIO.bits.inst := Mux(ciscValid, ciscInst.inst.asTypeOf(gemminiIO.bits.inst), regCommand)
  gemminiIO.bits.rs1 := Mux(ciscValid, ciscInst.rs1, Cat(gemminiRs1RegMSB, gemminiRs1RegLSB))
  gemminiIO.bits.rs2 := Mux(ciscValid, ciscInst.rs2, Cat(gemminiRs2RegMSB, gemminiRs2RegLSB))
  gemminiIO.valid := ciscValid || regValid
  assert(gemminiIO.ready || !gemminiIO.valid)

  accSlave.status := RegNext(outer.gemmini.module.io.busy).asUInt

  outer.traceSourceNode.bundle := DontCare
  outer.traceSourceNode.bundle.insns foreach (_.valid := false.B)

  // hacky, but cluster will AND the cease signals from all tiles, and we want
  // the core tiles to determine cluster cease not Gemmini
  outer.reportCease(Some(true.B))
}
