package radiance.memory

import chisel3._
import chisel3.util._
import chisel3.experimental._
import freechips.rocketchip.diplomacy._
import org.chipsalliance.diplomacy.lazymodule.{LazyModule, LazyModuleImp}
import freechips.rocketchip.tilelink._
import org.chipsalliance.cde.config.{Parameters, Field}

case object VortexL1Key extends Field[Option[VortexL1Config]](None /*default*/ )

case class VortexL1Config(
                           cacheSize: Int, // total cache size in bytes
                           numBanks: Int,
                           inputSize: Int, // This is the read/write granularity of the L1 cache
                           cacheLineSize: Int,
                           writeInfoReqQSize: Int,
                           mshrSize: Int,
                           memSideSourceIds: Int,
                           uncachedAddrSets: Seq[AddressSet]
) {
  // NOTE: This assertion depends on the fact that the Vortex cache is
  // configured to have 1 bank, and that it uses MSHR id as the tag of
  // memory-side requests.  Otherwise, it will append bank id to the tag as
  // well and break this requirement.
  require(
    mshrSize == memSideSourceIds,
    "MSHR size must match the number of sourceIds to downstream."
  )
}

object defaultVortexL1Config
    extends VortexL1Config(
      cacheSize = 16384,
      numBanks = 4,
      inputSize = 16,
      cacheLineSize = 16, // bytes
      writeInfoReqQSize = 16,
      mshrSize = 8,
      memSideSourceIds = 8,
      // Don't cache CLINT region to ensure coherent access
      uncachedAddrSets = Seq(AddressSet(0x2000000L, 0xffffL))
    )

class VortexL1Cache(config: VortexL1Config)(implicit p: Parameters)
    extends LazyModule {
  val banks = Seq.tabulate(config.numBanks) { bankId =>
    // helps with name mangling in Verilog
    val bank = LazyModule(new VortexBank(config, bankId))
    bank
  }
  // passthrough
  val passThrough = LazyModule(new VortexBankPassThrough(config))

  // visibility node that exposes to upstream
  val coresideNode = TLIdentityNode()

  // core-side crossbar that arbitrates core requests to banks
  protected val bankXbar = LazyModule(new TLXbar)
  bankXbar.node :=* coresideNode
  banks.foreach { _.coresideNode :=* bankXbar.node }
  passThrough.coresideNode :=* bankXbar.node

  // master node that exposes to and drives the downstream
  val masterNode = TLIdentityNode()
  banks.foreach { masterNode := _.vxCacheToL2Node }
  masterNode := passThrough.vxCacheToL2Node

  lazy val module = new LazyModuleImp(this)
}

// TODO: Make this a Blocking Module
class VortexBankPassThrough(config: VortexL1Config)(implicit p: Parameters)
    extends LazyModule {
  // Slave node to upstream
  val managerParam = Seq(
    TLSlavePortParameters.v1(
      beatBytes = config.inputSize,
      managers = Seq(
        TLSlaveParameters.v1(
          address = config.uncachedAddrSets,
          regionType = RegionType.IDEMPOTENT,
          executable = false,
          supportsGet = TransferSizes(1, config.inputSize),
          supportsPutPartial = TransferSizes(1, config.inputSize),
          supportsPutFull = TransferSizes(1, config.inputSize),
          fifoId = Some(0)
        )
      )
    )
  )

  // HACK: Set arbitrarily since we cannot query the coresideNode's sourceId
  // here. See comment on the require below.
  // @perf: This is quite high
  val sourceWidth = 9

  // Master node to downstream
  val clientParam = Seq(
    TLMasterPortParameters.v1(
      clients = Seq(
        TLMasterParameters.v1(
          name = "VortexBankPassthrough",
          sourceId = IdRange(0, 1 << sourceWidth),
          supportsProbe = TransferSizes(1, config.cacheLineSize),
          supportsGet = TransferSizes(1, config.cacheLineSize),
          supportsPutFull = TransferSizes(1, config.cacheLineSize),
          supportsPutPartial = TransferSizes(1, config.cacheLineSize)
        )
      )
    )
  )

  val coresideNode = TLManagerNode(managerParam)
  val vxCacheFetchNode = TLClientNode(clientParam)
  val vxCacheToL2Node = TLIdentityNode()
  vxCacheToL2Node := TLWidthWidget(config.cacheLineSize) := vxCacheFetchNode

  // passthrough logic
  lazy val module = new LazyModuleImp(this) {
    val (upstream, _) = coresideNode.in(0)
    val (downstream, _) = vxCacheFetchNode.out(0)

    // Make sure the outgoing edge of this passthrough has enough sourceIds
    // that encompasses the core-side incoming edge's.  This is an unfortunate
    // hack due to incomplete param negotiations across disconnected Diplomacy
    // graphs.
    // println(s"${upstream.params.sourceBits} <= ${downstream.params.sourceBits}")
    require(upstream.params.sourceBits <= downstream.params.sourceBits,
            "mem-side source of L1 cache truncates core-side source! " +
            "Try lowering core/coalescer srcIds, or increasing sourceWidth " +
            "for VortexBankPassThrough")

    downstream.a <> upstream.a
    upstream.d <> downstream.d
  }
}

class VortexBank(
    config: VortexL1Config,
    bankId: Int,
)(implicit p: Parameters)
    extends LazyModule {
  // Generate AddressSet by excluding Addr we don't want
  def generateAddressSets(): Seq[AddressSet] = {
    // suppose have 4 bank
    // base for bank 1: ...000000|01|0000
    // mask for bank 1;    111111|00|1111
    val base = 0x00000000L | (bankId * config.inputSize)
    val mask = 0xffffffffL ^ ((config.numBanks - 1) * config.inputSize)

    val excludeSets = config.uncachedAddrSets
    var remainingSets: Seq[AddressSet] = Seq(AddressSet(base, mask))
    for (excludeSet <- excludeSets) {
      remainingSets = remainingSets.flatMap(_.subtract(excludeSet))
    }
    remainingSets
  }

  // Slave node to upstream
  val managerParam = Seq(
    TLSlavePortParameters.v1(
      beatBytes = config.inputSize,
      managers = Seq(
        TLSlaveParameters.v1(
          address = generateAddressSets(),
          regionType = RegionType.IDEMPOTENT, // idk what this does
          executable = false,
          supportsGet = TransferSizes(1, config.inputSize),
          supportsPutPartial = TransferSizes(1, config.inputSize),
          supportsPutFull = TransferSizes(1, config.inputSize),
          fifoId = Some(0)
        )
      )
    )
  )

  // Master node to downstream
  val clientParam = Seq(
    TLMasterPortParameters.v1(
      clients = Seq(
        TLMasterParameters.v1(
          name = s"VortexBank${bankId}",
          sourceId = IdRange(0, config.memSideSourceIds),
          supportsProbe = TransferSizes(1, config.inputSize),
          supportsGet = TransferSizes(1, config.inputSize),
          supportsPutFull = TransferSizes(1, config.inputSize),
          supportsPutPartial = TransferSizes(1, config.inputSize)
        )
      )
    )
  )

  // Core -> VxCache
  val coresideNode = TLManagerNode(managerParam)
  val vxCacheToL2Node = TLIdentityNode()
  val vxCacheFetchNode = TLClientNode(clientParam)

  // We need this widthWidget here, because whenever the bank is performing
  // read and write to Mem, it must have the illusion that dataWidth is as big
  // as as its cacheline size
  vxCacheToL2Node := TLWidthWidget(config.cacheLineSize) := vxCacheFetchNode
  lazy val module = new VortexBankImp(this, config);
}

class VortexBankImp(
    outer: VortexBank,
    config: VortexL1Config
) extends LazyModuleImp(outer) {
  val (tlInFromCoal, _) = outer.coresideNode.in.head
  val coreTagWidth = tlInFromCoal.a.bits.source.getWidth
  val coreTagWidthPlusSize = coreTagWidth + log2Ceil(config.inputSize)

  val vxCache = Module(
    new VX_cache_top(
      WORD_SIZE = config.inputSize,
      // distribute total size across numBanks
      CACHE_SIZE = config.cacheSize / config.numBanks,
      CACHE_LINE_SIZE = config.cacheLineSize,
      CORE_TAG_WIDTH = coreTagWidthPlusSize,
      MSHR_SIZE = config.mshrSize
    )
  );

  vxCache.io.clk := clock
  vxCache.io.reset := reset

  val writeReqCount = RegInit(UInt(32.W), 0.U)
  val writeInputFire = Wire(Bool())
  val writeOutputFire = Wire(Bool())

  when(writeInputFire && ~writeOutputFire) {
    writeReqCount := writeReqCount + 1.U
  }.elsewhen(~writeInputFire && writeOutputFire) {
    writeReqCount := writeReqCount - 1.U
  }

  dontTouch(writeInputFire)
  dontTouch(writeOutputFire)
  dontTouch(writeReqCount)

  class WriteReqInfo extends Bundle {
    val id = UInt(32.W)
    val size = UInt(32.W)
  }

  class ReadReqInfo(config: VortexL1Config) extends Bundle {
    val size = UInt(log2Ceil(4).W + 1)
    val id = UInt(coreTagWidth.W)
  }

  val coreWriteReqQueue = Module(
    new Queue(
      (new WriteReqInfo).cloneType,
      config.writeInfoReqQSize,
      true,
      false
    )
  )
  val readReqInfo = Wire(new ReadReqInfo(config))

  // Translate TL request from Coalescer to requests for VX_cache
  def TLReq2VXReq = {
    // coal -> vxCache
    tlInFromCoal.a.ready :=
      vxCache.io.core_req_ready && coreWriteReqQueue.io.enq.ready // not optimal
    vxCache.io.core_req_valid := tlInFromCoal.a.valid

    // read = 0, write = 1
    vxCache.io.core_req_rw := !(tlInFromCoal.a.bits.opcode === TLMessages.Get)
    // 4 is also hardcoded, it should be log2WordSize
    vxCache.io.core_req_addr := tlInFromCoal.a.bits.address(
      31,
      log2Ceil(config.inputSize)
    )
    vxCache.io.core_req_byteen := tlInFromCoal.a.bits.mask
    vxCache.io.core_req_data := tlInFromCoal.a.bits.data

    // combine size and tag field into one big wire, to put into
    // vxCache.io.core_req_tag
    readReqInfo.id := tlInFromCoal.a.bits.source
    readReqInfo.size := tlInFromCoal.a.bits.size
    assert(readReqInfo.id.getWidth == tlInFromCoal.a.bits.source.getWidth,
      s"id width mismatch; core-side ${tlInFromCoal.a.bits.source.getWidth}, cache-side ${readReqInfo.id.getWidth}")
    assert(readReqInfo.size.getWidth == tlInFromCoal.a.bits.size.getWidth,
      s"size width mismatch; core-side ${tlInFromCoal.a.bits.size.getWidth}, cache-side ${readReqInfo.size.getWidth}")
    // ignore param, size, corrupt
    vxCache.io.core_req_tag := readReqInfo.asTypeOf(vxCache.io.core_req_tag)

    writeInputFire := vxCache.io.core_req_rw && tlInFromCoal.a.fire

    // vxCache -> coal response on channel D
    //
    // Vortex L1 does not send back write responses that are required by
    // TileLink. Therefore we synthesize write responses here outside of L1 by
    // using a separate queue structure that keeps track of in-flight write
    // requests, and tries to send back the response as soon as the queue has
    // valid entries.
    //
    // We cannot assume that we can send back write responses at the next cycle
    // after the requests, since there can exist a contemporary read response
    // at the same cycle.

    assert(coreWriteReqQueue.io.enq.ready === true.B,
      "FIXME: VortexCache: coreWriteReqQueue is full")
    coreWriteReqQueue.io.enq.valid :=
      tlInFromCoal.a.fire && !(tlInFromCoal.a.bits.opcode === TLMessages.Get)
    coreWriteReqQueue.io.enq.bits.id := tlInFromCoal.a.bits.source
    coreWriteReqQueue.io.enq.bits.size := tlInFromCoal.a.bits.size

    // Prioritize ack for any pending reads over write acks in the queue. Don't
    // ack write if vxCache has a current valid response for reads (vxCache
    // response is always for reads.)
    coreWriteReqQueue.io.deq.ready := tlInFromCoal.d.ready && ~vxCache.io.core_rsp_valid

    // handle competition between a pending read ack response and write ack
    // response
    vxCache.io.core_rsp_ready := tlInFromCoal.d.ready
    tlInFromCoal.d.valid := vxCache.io.core_rsp_valid || coreWriteReqQueue.io.deq.valid
    tlInFromCoal.d.bits.source := Mux(
      vxCache.io.core_rsp_valid,
      vxCache.io.core_rsp_tag.asTypeOf(readReqInfo).id,
      coreWriteReqQueue.io.deq.bits.id
    )
    tlInFromCoal.d.bits.opcode := Mux(
      vxCache.io.core_rsp_valid, // always for reads
      TLMessages.AccessAckData,
      TLMessages.AccessAck
    )
    tlInFromCoal.d.bits.size := Mux(
      vxCache.io.core_rsp_valid,
      vxCache.io.core_rsp_tag.asTypeOf(readReqInfo).size,
      coreWriteReqQueue.io.deq.bits.size
    )
    tlInFromCoal.d.bits.param := 0.U
    tlInFromCoal.d.bits.sink := 0.U
    tlInFromCoal.d.bits.denied := false.B
    tlInFromCoal.d.bits.corrupt := false.B
    tlInFromCoal.d.bits.data := vxCache.io.core_rsp_data
  }

  // Since Vortex L1 is a write-through cache, it doesn't bookkeep writes in
  // its MSHR and therefore doesn't allocate a new tag id for write requests.
  // We use a separate source ID allocator to solve this.
  val sourceGen = Module(
    new SourceGenerator(
      log2Ceil(config.memSideSourceIds),
      metadata = Some(UInt(32.W)),
      ignoreInUse = false
    )
  )

  // Translate VX_cache mem request to a TL request to be sent to L2
  def VXReq2TLReq = {
    val (tlOutToL2, _) = outer.vxCacheFetchNode.out.head

    // vxCache -> downstream L2 request
    vxCache.io.mem_req_ready := tlOutToL2.a.ready && sourceGen.io.id.valid
    tlOutToL2.a.valid := vxCache.io.mem_req_valid && sourceGen.io.id.valid

    sourceGen.io.gen := tlOutToL2.a.fire
    sourceGen.io.meta := vxCache.io.mem_req_tag // save the old read id

    writeOutputFire := tlOutToL2.a.fire && vxCache.io.mem_req_rw

    tlOutToL2.a.bits.opcode := Mux(
      vxCache.io.mem_req_rw,
      Mux(
        vxCache.io.mem_req_byteen.andR,
        TLMessages.PutFullData,
        TLMessages.PutPartialData
      ),
      TLMessages.Get
    )

    tlOutToL2.a.bits.address := Cat(vxCache.io.mem_req_addr, 0.U(log2Ceil(config.cacheLineSize).W))
    tlOutToL2.a.bits.mask := Mux(
      vxCache.io.mem_req_rw,
      vxCache.io.mem_req_byteen,
      ~(0.U(config.cacheLineSize.W))
    )
    tlOutToL2.a.bits.data := vxCache.io.mem_req_data
    tlOutToL2.a.bits.source := sourceGen.io.id.bits
    // ignore param, size, corrupt fields
    tlOutToL2.a.bits.param := 0.U
    tlOutToL2.a.bits.size := log2Ceil(config.cacheLineSize).U
    tlOutToL2.a.bits.corrupt := false.B
    // downstream L2 -> vxCache response
    tlOutToL2.d.ready := vxCache.io.mem_rsp_ready

    vxCache.io.mem_rsp_valid :=
      tlOutToL2.d.valid && (tlOutToL2.d.bits.opcode === TLMessages.AccessAckData)
    vxCache.io.mem_rsp_tag := sourceGen.io.peek
    vxCache.io.mem_rsp_data := tlOutToL2.d.bits.data

    sourceGen.io.reclaim.valid := tlOutToL2.d.fire
    sourceGen.io.reclaim.bits := tlOutToL2.d.bits.source
  }

  TLReq2VXReq
  VXReq2TLReq
}

class VX_cache_top(
    // these values should match the default settings in Verilog
    // TODO: INSTANCE_ID
    CACHE_SIZE: Int = 16384,
    CACHE_LINE_SIZE: Int = 16,
    NUM_WAYS: Int = 4,
    // for single-bank configuration, set NUM_REQS = 1 and instead set
    // WORD_SIZE to something wider than 4
    WORD_SIZE: Int = 16,
    CRSQ_SIZE: Int = 2,
    MSHR_SIZE: Int = 8,
    MRSQ_SIZE: Int = 0,
    MREQ_SIZE: Int = 4,
    WRITE_ENABLE: Int = 1,
    UUID_WIDTH: Int = 0, // FIXME: should be different for debug
    CORE_TAG_WIDTH: Int =
      16, // source ID ranges from 0 to 1 << 10, we need to allocate upper bits to save size
    CORE_OUT_REG : Int = 0,
    MEM_OUT_REG : Int = 0,
) extends BlackBox(
      Map(
        // NOTE: NUM_REQS is analogous to SIMD width, whereas NUM_BANKS is the
        // actual number of banks.  In the original Vortex code, VX_cache has
        // VX_stream_xbar that arbitrates the incoming NUM_REQS into outgoing
        // NUM_BANKS.  Since we do that logic ourselves using TL Xbars, fix
        // those params to 1 for Verilog.
        "NUM_REQS" -> 1,
        "CACHE_SIZE" -> CACHE_SIZE,
        "LINE_SIZE" -> CACHE_LINE_SIZE,
        // NUM_BANKS is set to 1 to treat a whole VX_cache_top instance as a
        // single bank
        "NUM_BANKS" -> 1,
        "NUM_WAYS" -> NUM_WAYS,
        "WORD_SIZE" -> WORD_SIZE,
        "CRSQ_SIZE" -> CRSQ_SIZE,
        "MSHR_SIZE" -> MSHR_SIZE,
        "MRSQ_SIZE" -> MRSQ_SIZE,
        "MREQ_SIZE" -> MREQ_SIZE,
        "WRITE_ENABLE" -> WRITE_ENABLE,
        "UUID_WIDTH" -> UUID_WIDTH,
        "TAG_WIDTH" -> CORE_TAG_WIDTH,
        "CORE_OUT_REG" -> CORE_OUT_REG,
        "MEM_OUT_REG" -> MEM_OUT_REG,
        // Although VX_cache_top exposes it as a parameter, MEM_TAG_WIDTH is
        // not really configurable -- it is set to be a concatenation of the
        // MSHR id and cache bank id.  Instead of trying to configure it from
        // Chisel side, we try to figure out its value that's elaborated in the
        // Verilog side and configure the Chisel io width correspondingly.
        // "MEM_TAG_WIDTH" -> MEM_TAG_WIDTH
      )
    )
    with HasBlackBoxResource {

  def memTagWidth(mshrSize: Int, numBanks: Int): Int =
    log2Ceil(mshrSize) + log2Ceil(numBanks)
  val MEM_TAG_WIDTH = memTagWidth(MSHR_SIZE, 1/* NUM_BANKS */)

  // These logic is fixed in VX_cache_define.vh
  val memAddrWidth = 32 // FIXME hardcoded
  val cacheWordAddrWidth = 32 - log2Ceil(WORD_SIZE)
  val cacheMemAddrWidth = 32 - log2Ceil(CACHE_LINE_SIZE)

  val io = IO(new Bundle {
    val clk = Input(Clock())
    val reset = Input(Reset())

    // CACHE <> CORE
    val core_req_valid = Input(Bool())
    val core_req_rw = Input(Bool())
    val core_req_byteen = Input(UInt(WORD_SIZE.W))
    val core_req_addr = Input(UInt(cacheWordAddrWidth.W))
    val core_req_data = Input(UInt((WORD_SIZE * 8).W))
    val core_req_tag = Input(UInt(CORE_TAG_WIDTH.W))
    val core_req_ready = Output(Bool())

    val core_rsp_valid = Output(Bool()) // 1 bit wide
    val core_rsp_data = Output(UInt((WORD_SIZE * 8).W))
    val core_rsp_tag = Output(UInt(CORE_TAG_WIDTH.W))
    val core_rsp_ready = Input(Bool())

    // CACHE <> L2
    val mem_req_valid = Output(Bool())
    val mem_req_rw = Output(Bool())
    val mem_req_byteen = Output(UInt(CACHE_LINE_SIZE.W))
    val mem_req_addr = Output(UInt(cacheMemAddrWidth.W))
    val mem_req_data = Output(UInt((CACHE_LINE_SIZE * 8).W))
    val mem_req_tag = Output(UInt(MEM_TAG_WIDTH.W))
    val mem_req_ready = Input(Bool())

    val mem_rsp_valid = Input(Bool())
    val mem_rsp_data = Input(UInt((CACHE_LINE_SIZE * 8).W))
    val mem_rsp_tag = Input(UInt(MEM_TAG_WIDTH.W))
    val mem_rsp_ready = Output(Bool())
  })

  addResource("/vsrc/vortex/hw/rtl/cache/VX_cache_bank.sv")
  // addResource("/vsrc/vortex/hw/rtl/cache/VX_cache_bypass.sv")
  addResource("/vsrc/vortex/hw/rtl/cache/VX_cache_data.sv")
  addResource("/vsrc/vortex/hw/rtl/cache/VX_cache_define.vh")
  addResource("/vsrc/vortex/hw/rtl/cache/VX_cache_init.sv")
  addResource("/vsrc/vortex/hw/rtl/cache/VX_cache_mshr.sv")
  addResource("/vsrc/vortex/hw/rtl/cache/VX_cache.sv")
  addResource("/vsrc/vortex/hw/rtl/cache/VX_cache_tags.sv")
  addResource("/vsrc/vortex/hw/rtl/cache/VX_cache_top.sv")
}

// <FIXME> Delete the following NewSourceGenerator when merging with origin/graphics
// we should just use the one in coalescing.scala written by hansung

class NewSourceGenerator[T <: Data](
    sourceWidth: Int,
    metadata: Option[T] = None,
    ignoreInUse: Boolean = false
) extends Module {
  def getMetadataType = metadata match {
    case Some(gen) => gen.cloneType
    case None      => UInt(0.W)
  }
  val io = IO(new Bundle {
    val gen = Input(Bool())
    val reclaim = Input(Valid(UInt(sourceWidth.W)))
    val id = Output(Valid(UInt(sourceWidth.W)))
    // below are used only when metadata is not None
    // `meta` is used as input when a request succeeds id generation to store
    // its value to the table.
    // `peek` is the retrieved metadata saved for the request when corresponding
    // request has come back, setting `reclaim`.
    // Although these do not use ValidIO, it is safe because any in-flight
    // response coming back should have allocated a valid entry in the table
    // when it went out.
    val meta = Input(getMetadataType)
    val peek = Output(getMetadataType)
    // for debugging; indicates whether there is at least one inflight request
    // that hasn't been reclaimed yet
    val inflight = Output(Bool())
  })
  val head = RegInit(UInt(sourceWidth.W), 0.U)
  head := Mux(io.gen, head + 1.U, head)

  val outstanding = RegInit(UInt((sourceWidth + 1).W), 0.U)
  io.inflight := (outstanding > 0.U) || io.gen

  val numSourceId = 1 << sourceWidth
  val row = new Bundle {
    val meta = getMetadataType
    val id = Valid(UInt(sourceWidth.W))
    val age = UInt(32.W) // New age field for debugging
  }
  // valid: in use, invalid: available
  // val occupancyTable = Mem(numSourceId, Valid(UInt(sourceWidth.W)))
  val occupancyTable = Mem(numSourceId, row)
  when(reset.asBool) {
    (0 until numSourceId).foreach { i =>
      occupancyTable(i).id.valid := false.B
      occupancyTable(i).meta := 0.U
      occupancyTable(i).age := 0.U
    }
  }
  val frees = (0 until numSourceId).map(!occupancyTable(_).id.valid)
  val lowestFree = PriorityEncoder(frees)
  val lowestFreeRow = occupancyTable(lowestFree)

  io.id.valid := (if (ignoreInUse) true.B else !lowestFreeRow.id.valid)
  io.id.bits := lowestFree
  when(io.gen && io.id.valid /* fire */ ) {
    occupancyTable(io.id.bits).id.valid := true.B // mark in use
    occupancyTable(
      io.id.bits
    ).age := 0.U // reset age upon issuing, double safety
    if (metadata.isDefined) {
      occupancyTable(io.id.bits).meta := io.meta
    }
  }

  // Increase age of all inflight IDs by 1, except for the one being reclaimed
  for (i <- 0 until numSourceId) {
    when(
      occupancyTable(
        i
      ).id.valid && (i.U =/= io.reclaim.bits || !io.reclaim.valid)
    ) {
      occupancyTable(i).age := occupancyTable(i).age + 1.U
    }
  }

  when(io.reclaim.valid) {
    assert(
      occupancyTable(io.reclaim.bits).id.valid === true.B,
      "tried to reclaim a non-used id"
    )
    occupancyTable(io.reclaim.bits).id.valid := false.B // mark freed
    occupancyTable(io.reclaim.bits).age := 0.U
  }

  io.peek := {
    if (metadata.isDefined) occupancyTable(io.reclaim.bits).meta else 0.U
  }

  when(io.gen && io.id.valid) {
    when(!io.reclaim.valid) {
      assert(outstanding < (1 << sourceWidth).U)
      outstanding := outstanding + 1.U
    }
  }.elsewhen(io.reclaim.valid) {
    assert(outstanding > 0.U)
    outstanding := outstanding - 1.U
  }

  // Debugging wires
  val ages = VecInit((0 until numSourceId).map(i => occupancyTable(i).age))
  val oldestIndex = PriorityEncoder(
    ages.map(a => a === ages.reduce((x, y) => Mux(x > y, x, y)))
  )
  val oldestIdInflight = Wire(UInt(sourceWidth.W))
  val oldestMetadata = Wire(getMetadataType)
  val oldestAge = Wire(UInt(32.W))

  oldestIdInflight := oldestIndex
  oldestMetadata := occupancyTable(oldestIndex).meta
  oldestAge := occupancyTable(oldestIndex).age
  assert(
    oldestAge <= 10000.U,
    "One id in the SourceGen is not released for long time, potential bug !"
  )

  dontTouch(oldestIdInflight)
  dontTouch(oldestMetadata)
  dontTouch(oldestAge)
  dontTouch(outstanding)

}
