// See LICENSE.SiFive for license details.
// See LICENSE.Berkeley for license details.

package radiance.tile

import chisel3._
import chisel3.experimental.AffectsChiselPrefix
import chisel3.util._
import freechips.rocketchip.devices.tilelink._
import freechips.rocketchip.diplomacy._
import org.chipsalliance.diplomacy.lazymodule.LazyModule
import freechips.rocketchip.prci.{ClockCrossingType, ClockSinkParameters, RationalCrossing}
import freechips.rocketchip.regmapper.RegField
import freechips.rocketchip.resources.BigIntHexContext
import freechips.rocketchip.rocket._
import freechips.rocketchip.subsystem.HierarchicalElementCrossingParamsLike
import freechips.rocketchip.tile._
import freechips.rocketchip.tilelink._
import freechips.rocketchip.util._
import midas.targetutils.SynthesizePrintf
import org.chipsalliance.cde.config._
import radiance.core._
import radiance.memory._
import radiance.subsystem.{GPUMemParams, GPUMemory, RadianceSimArgs}

/** For determining radiance core id.  This may be different from
 *  RadianceTileParams.tileId, when a cluster contains non-core tiles */
case object NumRadianceCores extends Field[Int](0)

case class RadianceTileParams(
    core: VortexCoreParams = VortexCoreParams(),
    useVxCache: Boolean = false,
    icache: Option[ICacheParams] = None /* Some(ICacheParams()) */,
    dcache: Option[DCacheParams] = None /* Some(DCacheParams()) */,
    btb: Option[BTBParams] = None, // Some(BTBParams()),
    dataScratchpadBytes: Int = 0,
    name: Option[String] = Some("radiance_tile"),
    tileId: Int = 0,
    coreId: Int = 0,
    beuAddr: Option[BigInt] = None,
    blockerCtrlAddr: Option[BigInt] = None,
    clockSinkParams: ClockSinkParameters = ClockSinkParameters(),
    boundaryBuffers: Option[RocketTileBoundaryBufferParams] = None
) extends InstantiableTileParams[RadianceTile] {
  // TODO: want to use ICache/DCacheParams as well
  // require(icache.isDefined)
  // require(dcache.isDefined)

  def instantiate(
      crossing: HierarchicalElementCrossingParamsLike,
      lookup: LookupByHartIdImpl
  )(implicit
      p: Parameters
  ): RadianceTile = {
    new RadianceTile(this, crossing, lookup)
  }
  val baseName = name.getOrElse("radiance_tile")
  val uniqueName = s"${baseName}_$tileId"
}

// TODO: move to VortexCore
// RadianceTileParams extends TileParams which require a `core: CoreParams`
// field, so VortexCoreParams needs to extend from CoreParams as well,
// requiring all these fields to be initialized.  Most of this is unnecessary
// though. TODO see how BOOM does that
case class VortexCoreParams(
  bootFreqHz: BigInt = 0,
  useVM: Boolean = false,
  useUser: Boolean = false,
  useSupervisor: Boolean = false,
  useHypervisor: Boolean = false,
  useDebug: Boolean = true,
  useAtomics: Boolean = false,
  useAtomicsOnlyForIO: Boolean = false,
  useCompressed: Boolean = false,
  useRVE: Boolean = false,
  useConditionalZero: Boolean = false,
  nLocalInterrupts: Int = 0,
  useNMI: Boolean = false,
  nBreakpoints: Int = 1,
  useBPWatch: Boolean = false,
  mcontextWidth: Int = 0,
  scontextWidth: Int = 0,
  nPMPs: Int = 8,
  nPerfCounters: Int = 0,
  haveBasicCounters: Boolean = true,
  haveCFlush: Boolean = false,
  misaWritable: Boolean = true,
  nL2TLBEntries: Int = 0,
  nL2TLBWays: Int = 1,
  nPTECacheEntries: Int = 8,
  mtvecInit: Option[BigInt] = Some(BigInt(0)),
  mtvecWritable: Boolean = true,
  fastLoadWord: Boolean = true,
  fastLoadByte: Boolean = false,
  branchPredictionModeCSR: Boolean = false,
  clockGate: Boolean = false,
  mvendorid: Int = 0, // 0 means non-commercial implementation
  mimpid: Int = 0x20181004, // release date in BCD
  mulDiv: Option[MulDivParams] = None,
  fpu: Option[FPUParams] = None,
  tensorCoreFP16: Boolean = false, // FP16 if true, FP32 if false
  tensorCoreDecoupled: Boolean = false, // hopper-style SMEM operand decoupling
  debugROB: Boolean = false, // if enabled, uses a C++ debug ROB to generate trace-with-wdata
  haveCease: Boolean = true, // non-standard CEASE instruction
  haveSimTimeout: Boolean = true // add plusarg for simulation timeout
) extends CoreParams {
  val haveFSDirty = false
  val pmpGranularity: Int = if (useHypervisor) 4096 else 4
  val fetchWidth: Int = if (useCompressed) 2 else 1
  val decodeWidth: Int = fetchWidth / (if (useCompressed) 2 else 1)
  val retireWidth: Int = 1
  val instBits: Int = if (useCompressed) 16 else 32
  val lrscCycles: Int = 80 // worst case is 14 mispredicted branches + slop
  val traceHasWdata: Boolean = false // ooo wb, so no wdata in trace
}

class RadianceTile private (
    val radianceParams: RadianceTileParams,
    crossing: ClockCrossingType,
    lookup: LookupByHartIdImpl,
    q: Parameters
) extends BaseTile(radianceParams, crossing, lookup, q)
    with SinksExternalInterrupts
    with SourcesExternalNotifications {
  // Private constructor ensures altered LazyModule.p is used implicitly
  def this(
      params: RadianceTileParams,
      crossing: HierarchicalElementCrossingParamsLike,
      lookup: LookupByHartIdImpl
  )(implicit p: Parameters) =
    this(params, crossing.crossingType, lookup, p)

  val intOutwardNode = None
  val slaveNode = TLIdentityNode()
  val masterNode = visibilityNode

  // Memory-mapped region for HTIF communication
  // We use fixed addresses instead of tohost/fromhost
  /* val regDevice =
    new SimpleDevice("radiance-reg", Seq(s"radiance-reg${tileParams.tileId}"))
  val regNode = TLRegisterNode(
    address = Seq(AddressSet(0x7c000000 + 0x1000 * tileParams.tileId, 0xfff)),
    device = regDevice,
    beatBytes = 4,
    concurrency = 1
  )

  regNode := TLFragmenter(4, 64) := tlSlaveXbar.node */

  require(
    p(SIMTCoreKey).isDefined,
    "SIMTCoreKey not defined; make sure to use WithSimtConfig when using RadianceTile"
  )

  // NOTE: when changing these, remember to change +define+NUM_CORES/THREADS/WARPS in
  // radiance.mk as well!
  val numWarps = p(SIMTCoreKey) match {
    case Some(simtParam) => simtParam.nWarps
    case None            => 4
  }
  val numCoreLanes = p(SIMTCoreKey) match {
    case Some(simtParam) => simtParam.nCoreLanes
    case None            => 4
  }
  val numLsuLanes = p(SIMTCoreKey) match {
    case Some(simtParam) => simtParam.nMemLanes
    case None            => 4
  }

  // CAUTION: imemSourceWidth is dependent on the ibuffer size.  We have to make
  // sure (1 << imemSourceWidth) is smaller than the per-warp ibuffer size;
  // otherwise, more requests than what ibuffer can accommodate can fire, and
  // responses might stall in the downstream.  This might cause issues when
  // there is also an outstanding dmem response that gets blocked by a previous
  // imem response due to serialization at the single tile<->sbus port, leading
  // to a stall in the backend pipeline and resulting in a deadlock.
  val imemSourceWidth = 4 // 1 << imemSourceWidth == IBUF_SIZE

  val smemSourceWidth = p(SIMTCoreKey) match {
    case Some(simtParam) => log2Ceil(simtParam.nSrcIds)
    case None => 4
  }

  val dmemSourceWidth = p(CoalescerKey) match {
    case Some(coalParam) => log2Ceil(coalParam.numOldSrcIds)
    case None => smemSourceWidth
  }
  // require(
  //   dmemSourceWidth >= 4,
  //   "Setting a small number of sourceIds may cause correctness bug inside " +
  //     "Vortex core due to synchronization issues in vx_wspawn. " +
  //     "We recommend setting nSrcIds to at least 16."
  // )

  val tensorTagWidth = 4 // hardcoded

  // Replicates some of the logic of how Vortex determines the tag width of
  // memory requests so that Chisel and Verilog are in agreement on bitwidths.
  // See VX_gpu_pkg.sv
  val NW_WIDTH = (if (numWarps == 1) 1 else log2Ceil(numWarps))
  val UUID_WIDTH = p(RadianceSimArgs) match {
    case Some(true) => 44
    case Some(false) => 1
    case None => 1
  }
  val imemTagWidth = UUID_WIDTH + NW_WIDTH

  require(numWarps >= numLsuLanes,
        s"Vortex core requires numWarps (${numWarps}) >= numLsuLanes (${numLsuLanes})")
  val LSUQ_SIZE = p(SIMTCoreKey).get.nSrcIds
  val LSUQ_TAG_BITS = log2Ceil(LSUQ_SIZE) + 1 /*DCACHE_BATCH_SEL_BITS*/
  val dmemTagWidth = UUID_WIDTH + LSUQ_TAG_BITS
  // dmem and smem shares the same tag width, DCACHE_NOSM_TAG_WIDTH
  val smemTagWidth = dmemTagWidth

  val imemNodes = Seq.tabulate(1) { i =>
    TLClientNode(
      Seq(
        TLMasterPortParameters.v1(
          clients = Seq(
            TLMasterParameters.v1(
              sourceId = IdRange(0, 1 << imemSourceWidth),
              name = s"Vortex Core ${radianceParams.coreId} I-Mem $i",
              requestFifo = true,
              supportsProbe =
                TransferSizes(1, lazyCoreParamsView.coreDataBytes),
              supportsGet = TransferSizes(1, lazyCoreParamsView.coreDataBytes)
            )
          )
        )
      )
    )
  }

  val dmemNodes = Seq.tabulate(numLsuLanes) { i =>
    TLClientNode(
      Seq(
        TLMasterPortParameters.v1(
          clients = Seq(
            TLMasterParameters.v1(
              sourceId = IdRange(0, 1 << dmemSourceWidth),
              name = s"Vortex Core ${radianceParams.coreId} D-Mem Lane $i",
              requestFifo = true,
              supportsProbe =
                TransferSizes(1, lazyCoreParamsView.coreDataBytes),
              supportsGet = TransferSizes(1, lazyCoreParamsView.coreDataBytes),
              supportsPutFull =
                TransferSizes(1, lazyCoreParamsView.coreDataBytes),
              supportsPutPartial =
                TransferSizes(1, lazyCoreParamsView.coreDataBytes)
            )
          )
        )
      )
    )
  }

  val smemNodes = Seq.tabulate(numLsuLanes) { i =>
    TLClientNode(
      Seq(
        TLMasterPortParameters.v1(
          clients = Seq(
            TLMasterParameters.v1(
              sourceId = IdRange(0, 1 << smemSourceWidth),
              name = s"Vortex Core ${radianceParams.coreId} SharedMem Lane $i",
              requestFifo = true,
              supportsProbe =
                TransferSizes(1, lazyCoreParamsView.coreDataBytes),
              supportsGet = TransferSizes(1, lazyCoreParamsView.coreDataBytes),
              supportsPutFull =
                TransferSizes(1, lazyCoreParamsView.coreDataBytes),
              supportsPutPartial =
                TransferSizes(1, lazyCoreParamsView.coreDataBytes)
            )
          )
        )
      )
    )
  }

  val tcSmemSize = 32
  val tcSmemNodes = Seq.tabulate(if (radianceParams.core.tensorCoreDecoupled) 2 else 0) { i =>
    TLClientNode(Seq(TLMasterPortParameters.v2(
      masters = Seq(TLMasterParameters.v2(
        name = s"rad_tc_${radianceParams.coreId}_$i",
        sourceId = IdRange(0, 1 << smemSourceWidth),
        supports = TLSlaveToMasterTransferSizes(
          probe = TransferSizes(1, tcSmemSize),
          get = TransferSizes(1, tcSmemSize),
        ),
        requestFifo = true
      ))
    )))
  }

  // combine outgoing per-lane dmemNode into 1 idenity node
  //
  // NOTE: We need TLWidthWidget here because there might be a data width
  // mismatch between Vortex's per-lane response and the system bus when we
  // don't instantiate either L1 or the coalescer.  This _should_ be optimized
  // out when we instantiate either which should handle data width conversion
  // internally (which it does by... using TLWidthWidget).
  val dmemAggregateNode = TLIdentityNode()
  dmemNodes.foreach { dmemAggregateNode := TLWidthWidget(4) := _ }

  val memNode = TLClientNode(
    Seq(
      TLMasterPortParameters.v1(
        clients = Seq(
          TLMasterParameters.v1(
            // FIXME: need to also respect imemSourceWidth
            sourceId = IdRange(0, 1 << dmemSourceWidth),
            name = s"Vortex Core ${radianceParams.coreId} Mem Interface",
            requestFifo = true,
            supportsProbe = TransferSizes(16, 16), // FIXME: hardcoded
            supportsGet = TransferSizes(16, 16),
            supportsPutFull = TransferSizes(16, 16),
            supportsPutPartial = TransferSizes(16, 16)
          )
        )
      )
    )
  )

  // Conditionally instantiate memory coalescer
  val coalescerNode = p(CoalescerKey) match {
    case Some(coalParam) => {
      val coal = LazyModule(
        new CoalescingUnit(coalParam)
      )
      coal.cpuNode :=* dmemAggregateNode
      coal.aggregateNode // N+1 lanes
    }
    case None => dmemAggregateNode
  }

  // these are the nodes that the tile egress node (tlMasterXbar) sees at the
  // upstream core/cache side
  val (icacheNode, dcacheNode): (TLNode, TLNode) = p(VortexL1Key) match {
    case Some(vortexL1Config) => {
      println("VortexL1Cache instantiated")
      // require(
      //   p(CoalescerKey).isDefined,
      //   "Vortex L1 configuration currently only works when coalescer is also enabled."
      // )

      val icache = LazyModule(new VortexL1Cache(vortexL1Config.copy(
        numBanks = 1
      )))
      val dcache = LazyModule(new VortexL1Cache(vortexL1Config))
      assert(imemNodes.length == 1)
      icache.coresideNode := TLWidthWidget(4) := imemNodes(0)
      // dmemNodes go through coalescerNode
      dcache.coresideNode :=* coalescerNode
      (icache.masterNode, dcache.masterNode)
    }
    case None => {
      val imemWideNode = TLIdentityNode()
      assert(imemNodes.length == 1)
      imemWideNode := TLWidthWidget(4) := imemNodes(0)
      (imemWideNode, coalescerNode)
    }
  }

  // Barrier synchronization node
  // FIXME: hardcoded param eq
  val numBarriers = numWarps
  def barrierIdBits = log2Ceil(numBarriers)
  val barrierMasterNode = BarrierMasterNode(barrierIdBits)

  val accMasterNode = AccMasterNode()

  val base = p(GPUMemory()) match {
    case Some(GPUMemParams(baseAddr, _)) => baseAddr
    case _ => BigInt(0)
  }

  if (radianceParams.useVxCache) {
    tlMasterXbar.node := AddressOrNode(base) := TLWidthWidget(16) := memNode
  } else {
    // imemNodes.foreach { tlMasterXbar.node := TLWidthWidget(4) := _ }
    tlMasterXbar.node :=* AddressOrNode(base) :=* icacheNode
    tlMasterXbar.node :=* AddressOrNode(base) :=* dcacheNode
  }

  /* below are copied from rocket */

  val tile_master_blocker =
    tileParams.blockerCtrlAddr
      .map(
        BasicBusBlockerParams(_, xBytes, masterPortBeatBytes, deadlock = true)
      )
      .map(bp => LazyModule(new BasicBusBlocker(bp)))

  tile_master_blocker.foreach(lm => connectTLSlave(lm.controlNode, xBytes))

  // TODO: this doesn't block other masters, e.g. RoCCs
  tlOtherMastersNode := tile_master_blocker.map {
    _.node := tlMasterXbar.node
  } getOrElse { tlMasterXbar.node }
  masterNode :=* tlOtherMastersNode
  org.chipsalliance.diplomacy.DisableMonitors { implicit p => tlSlaveXbar.node :*= slaveNode }

  val dtimProperty =
    Nil // Seq(dmemDevice.asProperty).flatMap(p => Map("sifive,dtim" -> p))

  val itimProperty =
    Nil // frontend.icache.itimProperty.toSeq.flatMap(p => Map("sifive,itim" -> p))

  // missing bus_error_unit

  val cpuDevice: SimpleDevice = new SimpleDevice(
    "cpu",
    Seq(s"sifive,radiance${tileParams.tileId}", "riscv")
  ) {
    override def parent = Some(ResourceAnchors.cpus)
    override def describe(resources: ResourceBindings): Description = {
      val Description(name, mapping) = super.describe(resources)
      Description(
        name,
        mapping ++ cpuProperties ++ nextLevelCacheProperty
          ++ tileProperties ++ dtimProperty ++ itimProperty /*++ beuProperty*/
      )
    }
  }

  ResourceBinding {
    Resource(cpuDevice, "reg").bind(ResourceAddress(tileId))
  }

  override lazy val module = new RadianceTileModuleImp(this)

  override def makeMasterBoundaryBuffers(
      crossing: ClockCrossingType
  )(implicit p: Parameters) = (radianceParams.boundaryBuffers, crossing) match {
    case (Some(RocketTileBoundaryBufferParams(true)), _) => TLBuffer()
    case (Some(RocketTileBoundaryBufferParams(false)), _: RationalCrossing) =>
      TLBuffer(
        BufferParams.none,
        BufferParams.flow,
        BufferParams.none,
        BufferParams.flow,
        BufferParams(1)
      )
    case _ => TLBuffer(BufferParams.none)
  }

  override def makeSlaveBoundaryBuffers(
      crossing: ClockCrossingType
  )(implicit p: Parameters) = (radianceParams.boundaryBuffers, crossing) match {
    case (Some(RocketTileBoundaryBufferParams(true)), _) => TLBuffer()
    case (Some(RocketTileBoundaryBufferParams(false)), _: RationalCrossing) =>
      TLBuffer(
        BufferParams.flow,
        BufferParams.none,
        BufferParams.none,
        BufferParams.none,
        BufferParams.none
      )
    case _ => TLBuffer(BufferParams.none)
  }
}

class RadianceTileModuleImp(outer: RadianceTile)
    extends BaseTileModuleImp(outer) {
  Annotated.params(this, outer.radianceParams)

  auto.elements.foreach({case (name, _) => 
      println(s"======= RadianceTile.elements.name: ${name}")
  })

  val core = Module(new Vortex(outer)(outer.p))

  core.io.clock := clock
  core.io.reset := reset

  class TwoWayCounter(width: Int) extends AffectsChiselPrefix {
    val value = RegInit(0.U(width.W))
    value := value
    def inc(): Unit = { value := value + 1.U }
    def dec(): Unit = { value := value - 1.U }
  }

  val dmemCounters = outer.dmemNodes.map { _ => new TwoWayCounter(outer.dmemSourceWidth) }
  val smemCounters = outer.smemNodes.map { _ => new TwoWayCounter(outer.smemSourceWidth) }
  core.io.downstream_mem_busy := VecInit(dmemCounters.map(_.value =/= 0.U)).reduceTree(_ || _) ||
    VecInit(smemCounters.map(_.value =/= 0.U)).reduceTree(_ || _)

  // begin @copypaste from RocketTile ------------------------------------------

  // reset vector is connected in the Frontend to s2_pc
  core.io.reset_vector := DontCare

  // outer.regNode.regmap(
  //   0x00 -> Seq(RegField.r(32, core.io.finished))
  // )

  // Report when the tile has ceased to retire instructions
  outer.reportCease(Some(core.io.finished))

  outer.reportWFI(Some(core.io.wfi))

  outer.decodeCoreInterrupts(core.io.interrupts) // Decode the interrupt vector

  when (core.io.interrupts.msip && !RegNext(core.io.interrupts.msip)) {
    SynthesizePrintf(printf("interrupt\n"))
  }

  core.io.interrupts.nmi.foreach { nmi => nmi := outer.nmiSinkNode.get.bundle }

  // Pass through various external constants and reports that were bundle-bridged into the tile
  // outer.traceSourceNode.bundle <> core.io.trace
  core.io.traceStall := outer.traceAuxSinkNode.bundle.stall
  // outer.bpwatchSourceNode.bundle <> core.io.bpwatch

  // not necessary for Vortex as hartId is set via Verilog parameter
  // core.io.hartid := outer.hartIdSinkNode.bundle
  // require(core.io.hartid.getWidth >= outer.hartIdSinkNode.bundle.getWidth,
  //   s"core hartid wire (${core.io.hartid.getWidth}b) truncates external hartid wire (${outer.hartIdSinkNode.bundle.getWidth}b)")

  // end @copypaste from RocketTile --------------------------------------------

  // ---------------------------------------------
  // Translate Vortex memory interface to TileLink
  // ---------------------------------------------

  if (outer.radianceParams.useVxCache) {
    println(s"width of a channel data ${core.io.mem.get.a.bits.data.getWidth}")
    println(s"width of d channel data ${core.io.mem.get.d.bits.data.getWidth}")

    val memTLAdapter = Module(
      new VortexTLAdapter(
        outer.dmemSourceWidth,
        chiselTypeOf(core.io.mem.get.a.bits),
        chiselTypeOf(core.io.mem.get.d.bits),
        outer.memNode.out.head
      )
    )

    // connection: VortexBundle <--> VortexTLAdapter <--> TL memNode
    memTLAdapter.io.inReq <> core.io.mem.get.a
    core.io.mem.get.d <> memTLAdapter.io.inResp
    outer.memNode.out(0)._1.a <> memTLAdapter.io.outReq
    memTLAdapter.io.outResp <> outer.memNode.out(0)._1.d
  } else {
    def connectImem = {
      val imemTLAdapter = Module(
        new VortexTLAdapter(
          outer.imemSourceWidth,
          chiselTypeOf(core.io.imem.get(0).a.bits),
          chiselTypeOf(core.io.imem.get(0).d.bits),
          outer.imemNodes.head.out.head
        )
      )
      // TODO: make imemNodes not a vector
      imemTLAdapter.io.inReq <> core.io.imem.get(0).a
      core.io.imem.get(0).d <> imemTLAdapter.io.inResp

      performanceCounters(Seq(imemTLAdapter.io.inReq), Seq(imemTLAdapter.io.inResp),
        desc = s"core${outer.radianceParams.coreId}-imem")

      // now connect TL adapter output ports to outer.imemNode, which can
      // either be L1 cache or tile egress
      outer.imemNodes(0).out(0)._1.a <> imemTLAdapter.io.outReq
      imemTLAdapter.io.outResp <> outer.imemNodes(0).out(0)._1.d
    }

    def connectDmem = {
      // @perf: this would duplicate SourceGenerator table for every lane and eat
      // up some area
      val dmemTLBundles = outer.dmemNodes.map(_.out.head._1)
      val dmemTLAdapters = Seq.tabulate(outer.numLsuLanes) { _ =>
        Module(
          new VortexTLAdapter(
            outer.dmemSourceWidth,
            new VortexBundleA(tagWidth = outer.dmemTagWidth, dataWidth = 32),
            new VortexBundleD(tagWidth = outer.dmemTagWidth, dataWidth = 32),
            outer.dmemNodes(0).out.head
          )
        )
      }

      // Since the individual per-lane TL requests might come back out-of-sync between
      // the lanes, but Vortex core expects the per-lane responses to be synced,
      // we need to selectively fire responses that have the same source, and
      // delay others.
      //
      // In order to do that, we pick a source from one of the valid lanes using e.g.
      // an arbiter.  Then using the chosen source id, we
      // - lie to core that response is not valid if source doesn't match picked, and
      // - lie to downstream that core is not ready if source doesn't match picked.
      //
      // Note that we cannot do this filtering logic using TileLink source ID, because
      // we're allocating source for each lane independently.  In that case, it's
      // possible that lane 0's source matches lane 1/2/3's source by chance,
      // even when they originated from different warps.  Using Vortex's dcache req tag
      // solves this issue because they use a UUID that is unique across all requests
      // in the program.
      //
      // TODO: A cleaner solution would be to simply do a synchronized allocation
      // of a same source id for all lanes.
      val arb = Module(
        new RRArbiter(
          // FIXME: should really be source on D channel
          new VortexBundleA(
            tagWidth = outer.dmemTagWidth,
            dataWidth = 32
          ).source.cloneType,
          outer.numLsuLanes
        )
      )
      arb.io.out.ready := true.B
      val dmemBundles = dmemTLAdapters.map(_.io.inResp)
      (arb.io.in zip dmemBundles).foreach { case (arbIn, vxDmem) =>
        arbIn.valid := vxDmem.valid
        arbIn.bits := vxDmem.bits.source
      }
      val matchingSources = Wire(UInt(outer.numLsuLanes.W))
      matchingSources := dmemBundles
        .map(b =>
          // If there is no valid response pending across all lanes,
          // matchingSources should not filter out upstream ready signals, so
          // set it to all-1
          !arb.io.out.valid || (b.bits.source === arb.io.out.bits)
        )
        .asUInt

      // make connection:
      // VortexBundle <--> sourceId filter <--> VortexTLAdapter <--> dmemNodes
      //
      // Chisel doesn't support 2-D array in BlackBox interface to Verilog, so
      // need to flatten everything.
      dmemTLAdapters.zipWithIndex.foreach {
        case (tlAdapter, i) =>
          // tlAdapter.io.inReq <> coreMem.a
          tlAdapter.io.inReq.valid := core.io.dmem_a_valid(i)
          tlAdapter.io.inReq.bits.opcode := core.io.dmem_a_bits_opcode(3 * (i + 1) - 1, 3 * i)
          tlAdapter.io.inReq.bits.size := core.io.dmem_a_bits_size(4 * (i + 1) - 1, 4 * i)
          tlAdapter.io.inReq.bits.source := core.io.dmem_a_bits_source(outer.dmemTagWidth * (i + 1) - 1, outer.dmemTagWidth * i)
          tlAdapter.io.inReq.bits.address := core.io.dmem_a_bits_address(32 * (i + 1) - 1, 32 * i)
          tlAdapter.io.inReq.bits.mask := core.io.dmem_a_bits_mask(4 * (i + 1) - 1, 4 * i)
          tlAdapter.io.inReq.bits.data := core.io.dmem_a_bits_data(32 * (i + 1) - 1, 32 * i)
      }
      core.io.dmem_a_ready := dmemTLAdapters.map(_.io.inReq.ready).asUInt

      core.io.dmem_d_valid := dmemTLAdapters.map(_.io.inResp.valid).asUInt
      core.io.dmem_d_bits_opcode := dmemTLAdapters.map(_.io.inResp.bits.opcode).asUInt
      core.io.dmem_d_bits_size := dmemTLAdapters.map(_.io.inResp.bits.size).asUInt
      core.io.dmem_d_bits_source := dmemTLAdapters.map(_.io.inResp.bits.source).asUInt
      core.io.dmem_d_bits_data := dmemTLAdapters.map(_.io.inResp.bits.data).asUInt

      // override response channel with matchingSources
      val dmem_d_valid_vec = Wire(Vec(outer.numLsuLanes, Bool()))
      dmemTLAdapters.zipWithIndex.foreach {
        case (tlAdapter, i) =>
          dmem_d_valid_vec(i) := tlAdapter.io.inResp.valid && matchingSources(i)
          tlAdapter.io.inResp.ready := core.io.dmem_d_ready(i) && matchingSources(i)
      }
      core.io.dmem_d_valid := dmem_d_valid_vec.asUInt

      (dmemTLAdapters zip dmemCounters).foreach { case (a, c) =>
        when (a.io.inReq.fire && !a.io.inResp.fire) {
          c.inc()
        }.elsewhen (a.io.inResp.fire && !a.io.inReq.fire) {
          c.dec()
        }
      }

      performanceCounters(dmemTLAdapters.map(_.io.inReq), dmemTLAdapters.map(_.io.inResp),
        desc = s"core${outer.radianceParams.coreId}-dmem")

      // now connect TL adapter output ports to outer.dmemNodes, which can
      // either be L1 cache or tile egress
      (dmemTLAdapters zip dmemTLBundles) foreach { case (tlAdapter, tlOut) =>
        tlOut.a <> tlAdapter.io.outReq
        tlAdapter.io.outResp <> tlOut.d
      }

      outer.dmemAggregateNode.out.foreach { bo =>
        dontTouch(bo._1.a)
        dontTouch(bo._1.d)
      }
    }

    def connectSmem = {
      // @perf: this would duplicate SourceGenerator table for every lane and eat
      // up some area
      val smemTLBundles = outer.smemNodes.map(_.out.head._1)
      val smemTLAdapters = Seq.tabulate(outer.numLsuLanes) { _ =>
        Module(
          new VortexTLAdapter(
            outer.smemSourceWidth,
            new VortexBundleA(tagWidth = outer.smemTagWidth, dataWidth = 32),
            new VortexBundleD(tagWidth = outer.smemTagWidth, dataWidth = 32),
            outer.smemNodes.head.out.head
          )
        )
      }

      smemTLAdapters.zipWithIndex.foreach {
        case (tlAdapter, i) =>
          // tlAdapter.io.inReq <> coreMem.a
          tlAdapter.io.inReq.valid := core.io.smem_a_valid(i)
          tlAdapter.io.inReq.bits.opcode := core.io.smem_a_bits_opcode(3 * (i + 1) - 1, 3 * i)
          tlAdapter.io.inReq.bits.size := core.io.smem_a_bits_size(4 * (i + 1) - 1, 4 * i)
          tlAdapter.io.inReq.bits.source := core.io.smem_a_bits_source(outer.smemTagWidth * (i + 1) - 1, outer.smemTagWidth * i)
          tlAdapter.io.inReq.bits.address := core.io.smem_a_bits_address(32 * (i + 1) - 1, 32 * i)
          tlAdapter.io.inReq.bits.mask := core.io.smem_a_bits_mask(4 * (i + 1) - 1, 4 * i)
          tlAdapter.io.inReq.bits.data := core.io.smem_a_bits_data(32 * (i + 1) - 1, 32 * i)
      }
      core.io.smem_a_ready := smemTLAdapters.map(_.io.inReq.ready).asUInt

      core.io.smem_d_valid := smemTLAdapters.map(_.io.inResp.valid).asUInt
      core.io.smem_d_bits_opcode := smemTLAdapters.map(_.io.inResp.bits.opcode).asUInt
      core.io.smem_d_bits_size := smemTLAdapters.map(_.io.inResp.bits.size).asUInt
      core.io.smem_d_bits_source := smemTLAdapters.map(_.io.inResp.bits.source).asUInt
      core.io.smem_d_bits_data := smemTLAdapters.map(_.io.inResp.bits.data).asUInt
      smemTLAdapters.zipWithIndex.foreach {
        case (tlAdapter, i) =>
          tlAdapter.io.inResp.ready := core.io.smem_d_ready(i)
      }

      (smemTLAdapters zip smemCounters).foreach { case (a, c) =>
        when (a.io.inReq.fire && !a.io.inResp.fire) {
          c.inc()
        }.elsewhen (a.io.inResp.fire && !a.io.inReq.fire) {
          c.dec()
        }
      }

      performanceCounters(smemTLAdapters.map(_.io.inReq), smemTLAdapters.map(_.io.inResp),
        desc = s"core${outer.radianceParams.coreId}-smem")

      (smemTLAdapters zip smemTLBundles) foreach { case (tlAdapter, tlOut) =>
        tlOut.a <> tlAdapter.io.outReq
        tlAdapter.io.outResp <> tlOut.d
      }
    }

    def connectTensor = {
      if (outer.radianceParams.core.tensorCoreDecoupled) {
        val tcb0 = new {
          val addr = core.io.tc_a_bits_address(31, 0)
          val tag = core.io.tc_a_bits_tag(outer.tensorTagWidth - 1, 0)
          val aValid = core.io.tc_a_valid(0)
          val dReady = core.io.tc_d_ready(0)
        }
        val tcb1 = new {
          val addr = core.io.tc_a_bits_address(63, 32)
          val tag = core.io.tc_a_bits_tag(4 + outer.tensorTagWidth - 1, 4)
          val aValid = core.io.tc_a_valid(1)
          val dReady = core.io.tc_d_ready(1)
        }
        val tcBundles = Seq(tcb0, tcb1)
        val adapters = (outer.tcSmemNodes zip tcBundles).zipWithIndex.map { case ((node, bundle), i) =>
          val client = node.out.head
          val adapter = Module(
            new VortexTLAdapter(
              outer.smemSourceWidth,
              new VortexBundleA(tagWidth = outer.tensorTagWidth, dataWidth = 32 * 8),
              new VortexBundleD(tagWidth = outer.tensorTagWidth, dataWidth = 32 * 8),
              client
            )
          )
          require(adapter.io.inReq.bits.source.widthOption.get == bundle.tag.widthOption.get)
          require(adapter.io.inReq.bits.address.widthOption.get == bundle.addr.widthOption.get)
          adapter.io.inReq.bits <> DontCare
          adapter.io.inReq.valid := bundle.aValid
          adapter.io.inReq.bits.address := bundle.addr
          adapter.io.inReq.bits.source := bundle.tag
          adapter.io.inReq.bits.size := 5.U // 256 bits
          adapter.io.inReq.bits.opcode := TLMessages.Get
          adapter.io.inReq.bits.mask := x"ffffffff".U
          adapter.io.inResp.ready := bundle.dReady

          client._1.a <> adapter.io.outReq
          adapter.io.outResp <> client._1.d
          adapter
        }
        core.io.tc_a_ready := Cat(adapters.last.io.inReq.ready, adapters.head.io.inReq.ready)
        core.io.tc_d_valid := Cat(adapters.last.io.inResp.valid, adapters.head.io.inResp.valid)
        core.io.tc_d_bits_data := Cat(adapters.last.io.inResp.bits.data, adapters.head.io.inResp.bits.data)
        core.io.tc_d_bits_tag := Cat(adapters.last.io.inResp.bits.source, adapters.head.io.inResp.bits.source)
        require(core.io.tc_d_bits_data.widthOption.get == adapters.head.io.inResp.bits.data.widthOption.get * 2)
        require(core.io.tc_d_bits_tag.widthOption.get == adapters.head.io.inResp.bits.source.widthOption.get * 2)
      } else {
        core.io.tc_a_ready := false.B
        core.io.tc_d_valid := false.B
        core.io.tc_d_bits_data := DontCare
        core.io.tc_d_bits_tag := DontCare
      }
    }

    def connectBarrier = {
      require(outer.barrierMasterNode.out.length == 1)
      // FIXME: bits not flattened
      outer.barrierMasterNode.out(0)._1.req.valid := core.io.gbar_req_valid
      outer.barrierMasterNode.out(0)._1.req.bits.barrierId := core.io.gbar_req_id
      outer.barrierMasterNode.out(0)._1.req.bits.coreId := core.io.gbar_req_core_id
      core.io.gbar_req_ready := outer.barrierMasterNode.out(0)._1.req.ready

      core.io.gbar_rsp_valid := outer.barrierMasterNode.out(0)._1.resp.valid
      core.io.gbar_rsp_id := outer.barrierMasterNode.out(0)._1.resp.bits.barrierId
      // core doesn't have a resp.ready port
      outer.barrierMasterNode.out(0)._1.resp.ready := true.B
    }

    def connectAccelerator = {
      outer.accMasterNode.out.head._1.cmd.bits := core.io.acc_write_out
      outer.accMasterNode.out.head._1.cmd.valid := core.io.acc_write_en
      core.io.acc_read_in := outer.accMasterNode.out.head._1.status
    }

    def performanceCounters(reqBundles: Seq[DecoupledIO[VortexBundleA]],
                            respBundles: Seq[DecoupledIO[VortexBundleD]],
                            desc: String) = {
      val currentPendingReqs = RegInit(SInt(32.W), 0.S)
      val pendingReqsCumulative = RegInit(SInt(32.W), 0.S)
      val totalReqs = RegInit(UInt(32.W), 0.U)

      val reqFireCountPerCycle = Wire(UInt(32.W))
      val respFireCountPerCycle = Wire(UInt(32.W))
      val reqReadFires = reqBundles.map { b => b.fire && b.bits.opcode === 4.U /* Get */ }
      val respReadFires = respBundles.map { b => b.fire && b.bits.opcode === 1.U /* AccessAckData */}
      reqFireCountPerCycle := PopCount(reqReadFires)
      respFireCountPerCycle := PopCount(respReadFires)
      totalReqs := totalReqs + reqFireCountPerCycle

      val diffPendingReqs = reqFireCountPerCycle.asSInt - respFireCountPerCycle.asSInt
      currentPendingReqs := currentPendingReqs + diffPendingReqs
      pendingReqsCumulative := pendingReqsCumulative + currentPendingReqs

      val prevFinished = RegNext(core.io.finished)
      val justFinished = !prevFinished && core.io.finished
      when (justFinished) {
        printf(s"PERF: ${desc}: average request latency (cum_pending / total): %d / %d\n",
               pendingReqsCumulative, totalReqs)
      }

      dontTouch(totalReqs)
      dontTouch(diffPendingReqs)
      dontTouch(currentPendingReqs)
      dontTouch(pendingReqsCumulative)
    }

    connectImem
    connectDmem
    connectSmem
    connectTensor
    connectBarrier
    connectAccelerator
  }

  // TODO: generalize for useVxCache
  if (!outer.radianceParams.useVxCache) {}

  // Instantiate a fake tensor core module to force unique-ification of module
  // names in the Chisel-generated Verilog.  These should be left out for
  // synthesis runs, although it's likely they will be optimized-out with all
  // inputs tied to low.

  if (outer.radianceParams.core.tensorCoreDecoupled) {
    val tensorNumSourceIds = (1 << outer.tensorTagWidth)
    val tensor = Module(new radiance.core.TensorCoreDecoupled(
      8, 8, half = true, tensorNumSourceIds))
    tensor.io.initiate.valid := false.B
    tensor.io.initiate.bits := DontCare
    tensor.io.respA.valid := false.B
    tensor.io.respA.bits := DontCare
    tensor.io.respB.valid := false.B
    tensor.io.respB.bits := DontCare
    tensor.io.respC := DontCare
    tensor.io.reqA.ready := false.B
    tensor.io.reqB.ready := false.B
    tensor.io.writeback.ready := false.B
  } else {
    if (outer.radianceParams.core.tensorCoreFP16) {
      val dpu = Module(new radiance.core.TensorDotProductUnit(4, half = true))
      dpu.io.in.valid := false.B
      dpu.io.in.bits.a := DontCare
      dpu.io.in.bits.b := DontCare
      dpu.io.in.bits.c := DontCare
      dpu.io.stall := false.B
    } else {
      val dpu = Module(new radiance.core.TensorDotProductUnit(2, half = false))
      dpu.io.in.valid := false.B
      dpu.io.in.bits.a := DontCare
      dpu.io.in.bits.b := DontCare
      dpu.io.in.bits.c := DontCare
      dpu.io.stall := false.B
    }
  }

  // // RoCC
  // if (outer.roccs.size > 0) {
  //   val (respArb, cmdRouter) = {
  //     val respArb = Module(new RRArbiter(new RoCCResponse()(outer.p), outer.roccs.size))
  //     val cmdRouter = Module(new RoccCommandRouter(outer.roccs.map(_.opcodes))(outer.p))
  //     outer.roccs.zipWithIndex.foreach { case (rocc, i) =>
  //       // ptwPorts ++= rocc.module.io.ptw
  //       rocc.module.io.ptw <> DontCare
  //       rocc.module.io.mem <> DontCare
  //       rocc.module.io.cmd <> cmdRouter.io.out(i)
  //       respArb.io.in(i) <> Queue(rocc.module.io.resp)
  //     }
  //     // Create this FPU just for RoCC
  //     // val nFPUPorts = outer.roccs.filter(_.usesFPU).size
  //     val fp_rocc_ios = outer.roccs.map(_.module.io)
  //     fp_rocc_ios.map { io =>
  //       io.fpu_req.ready := false.B
  //       io.fpu_resp.valid := false.B
  //       io.fpu_resp.bits := DontCare
  //     }
  //     (respArb, cmdRouter)
  //   }

  //   cmdRouter.io.in <> DontCare
  //   outer.roccs.foreach(_.module.io.exception := DontCare)
  //   respArb.io.out <> DontCare
  // }
}

// Some @copypaste from CoalescerSourceGen.
class VortexTLAdapter(
    newSourceWidth: Int,
    inReqT: VortexBundleA,
    inRespT: VortexBundleD,
    outTL: (TLBundle, TLEdge)
) extends Module {
  val io = IO(new Bundle {
    // in/out means upstream/downstream
    val inReq = Flipped(Decoupled(inReqT))
    val outReq = chiselTypeOf(outTL._1.a)
    val inResp = Decoupled(inRespT)
    val outResp = chiselTypeOf(outTL._1.d)
  })
  val (bundle, edge) = outTL
  val sourceGen = Module(
    new SourceGenerator(
      newSourceWidth,
      Some(inReqT.source),
      ignoreInUse = false
    )
  )
  sourceGen.io.gen := io.outReq.fire // use up a source ID only when request is created
  sourceGen.io.reclaim.valid := io.outResp.fire
  sourceGen.io.reclaim.bits := io.outResp.bits.source
  sourceGen.io.meta := io.inReq.bits.source

  // io passthrough logic
  // TLBundleA <> VortexBundleA
  io.outReq.valid := io.inReq.valid
  io.outReq.bits.opcode := io.inReq.bits.opcode
  io.outReq.bits.param := 0.U
  io.outReq.bits.size := io.inReq.bits.size
  io.outReq.bits.source := io.inReq.bits.source
  io.outReq.bits.address := io.inReq.bits.address
  // Get requires contiguous mask; only copy core's potentially-partial mask
  // when writing
  io.outReq.bits.mask := Mux(
    edge.hasData(io.outReq.bits),
    io.inReq.bits.mask,
    // generate TL-correct mask
    edge.mask(io.inReq.bits.address, io.inReq.bits.size)
  )
  io.outReq.bits.data := io.inReq.bits.data
  io.outReq.bits.corrupt := 0.U
  io.inReq.ready := io.outReq.ready
  // VortexBundleD <> TLBundleD
  io.inResp.valid := io.outResp.valid
  io.inResp.bits.opcode := io.outResp.bits.opcode
  io.inResp.bits.size := io.outResp.bits.size
  io.inResp.bits.source := io.outResp.bits.source
  io.inResp.bits.data := io.outResp.bits.data
  io.outResp.ready := io.inResp.ready

  // "man-in-the-middle"
  io.inReq.ready := io.outReq.ready && sourceGen.io.id.valid
  io.outReq.valid := io.inReq.valid && sourceGen.io.id.valid
  io.outReq.bits.source := sourceGen.io.id.bits
  // translate upstream response back to its old sourceId
  io.inResp.bits.source := sourceGen.io.peek
}
