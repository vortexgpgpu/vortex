// See LICENSE.SiFive for license details.
// See LICENSE.Berkeley for license details.

package radiance.subsystem

import chisel3._
import chisel3.util._
import org.chipsalliance.cde.config._
import freechips.rocketchip.rocket._
import freechips.rocketchip.tile._
import freechips.rocketchip.subsystem._
import gemmini._
import gemmini.Arithmetic.FloatArithmetic._
import radiance.tile._
import radiance.core._
import radiance.memory._
import radiance.subsystem.RadianceGemminiDataType.{BF16, FP16, FP32, Int8}

sealed trait RadianceSmemSerialization
case object FullySerialized extends RadianceSmemSerialization
case object CoreSerialized extends RadianceSmemSerialization
case object NotSerialized extends RadianceSmemSerialization

sealed trait MemType
case object TwoPort extends MemType
case object TwoReadOneWrite extends MemType

case class RadianceSharedMemKey(address: BigInt,
                                size: Int,
                                numBanks: Int,
                                numWords: Int,
                                wordSize: Int = 4,
                                memType: MemType = TwoPort,
                                strideByWord: Boolean = true,
                                filterAligned: Boolean = true,
                                disableMonitors: Boolean = true,
                                serializeUnaligned: RadianceSmemSerialization = FullySerialized)
case object RadianceSharedMemKey extends Field[Option[RadianceSharedMemKey]](None)

case class RadianceFrameBufferKey(baseAddress: BigInt,
                                  width: Int,
                                  size: Int,
                                  validAddress: BigInt,
                                  fbName: String = "fb")
case object RadianceFrameBufferKey extends Field[Seq[RadianceFrameBufferKey]](Seq())

class WithRadianceCores(
  n: Int,
  location: HierarchicalLocation,
  crossing: RocketCrossingParams,
  tensorCoreFP16: Boolean,
  tensorCoreDecoupled: Boolean,
  useVxCache: Boolean
) extends Config((site, _, up) => {
  case TilesLocated(`location`) => {
    val prev = up(TilesLocated(`location`))
    val idOffset = up(NumTiles)
    val coreIdOffset = up(NumRadianceCores)
    val vortex = RadianceTileParams(
      core = VortexCoreParams(
        tensorCoreFP16 = tensorCoreFP16,
        tensorCoreDecoupled = tensorCoreDecoupled
      ),
      btb = None,
      useVxCache = useVxCache,
      dcache = Some(DCacheParams(
        rowBits = site(SystemBusKey).beatBits,
        nSets = 64,
        nWays = 1,
        nTLBSets = 1,
        nTLBWays = 1,
        nTLBBasePageSectors = 1,
        nTLBSuperpages = 1,
        nMSHRs = 0,
        blockBytes = site(CacheBlockBytes))),
      icache = Some(ICacheParams(
        rowBits = site(SystemBusKey).beatBits,
        nSets = 64,
        nWays = 1,
        nTLBSets = 1,
        nTLBWays = 1,
        nTLBBasePageSectors = 1,
        nTLBSuperpages = 1,
        blockBytes = site(CacheBlockBytes))))
    List.tabulate(n)(i => RadianceTileAttachParams(
      vortex.copy(
        tileId = i + idOffset,
        coreId = i + coreIdOffset,
      ),
      crossing
    )) ++ prev
  }
  case NumTiles => up(NumTiles) + n
  case NumRadianceCores => up(NumRadianceCores) + n
}) {
  // constructor override that omits `crossing`
  def this(n: Int, location: HierarchicalLocation = InSubsystem,
    tensorCoreFP16: Boolean = false, tensorCoreDecoupled: Boolean = false,
    useVxCache: Boolean = false)
  = this(n, location, RocketCrossingParams(
    master = HierarchicalElementMasterPortParams.locationDefault(location),
    slave = HierarchicalElementSlavePortParams.locationDefault(location),
    mmioBaseAddressPrefixWhere = location match {
      case InSubsystem => CBUS
      case InCluster(clusterId) => CCBUS(clusterId)
    }
  ), tensorCoreFP16, tensorCoreDecoupled, useVxCache)
}

class WithEmulatorCores(
  n: Int,
  useVxCache: Boolean
) extends Config((site, _, up) => {
  case TilesLocated(InSubsystem) => {
    val prev = up(TilesLocated(InSubsystem))
    val idOffset = up(NumTiles)
    val emulator = EmulatorTileParams(
      core = VortexCoreParams(),
      useVxCache = useVxCache)
    List.tabulate(n)(i => EmulatorTileAttachParams(
      emulator.copy(tileId = i + idOffset),
      RocketCrossingParams()
    )) ++ prev
  }
  case NumTiles => up(NumTiles) + 1
  case NumRadianceCores => up(NumRadianceCores) + 1
})

class WithFuzzerCores(
  n: Int,
  useVxCache: Boolean
) extends Config((site, _, up) => {
  case TilesLocated(InSubsystem) => {
    val prev = up(TilesLocated(InSubsystem))
    val idOffset = up(NumTiles)
    val fuzzer = FuzzerTileParams(
      core = VortexCoreParams(),
      useVxCache = useVxCache)
    List.tabulate(n)(i => FuzzerTileAttachParams(
      fuzzer.copy(tileId = i + idOffset),
      RocketCrossingParams()
    )) ++ prev
  }
  case NumTiles => up(NumTiles) + 1
  case NumRadianceCores => up(NumRadianceCores) + 1
})

object RadianceGemminiDataType extends Enumeration {
  type Type = Value
  val FP32, FP16, BF16, Int8 = Value
}

class WithRadianceGemmini(location: HierarchicalLocation, crossing: RocketCrossingParams,
                          dim: Int, accSizeInKB: Int, tileSize: Either[(Int, Int, Int), Int],
                          dataType: RadianceGemminiDataType.Type, dmaBytes: Int) extends Config((site, _, up) => {
  case TilesLocated(`location`) => {
    val prev = up(TilesLocated(`location`))
    val idOffset = up(NumTiles)
    if (idOffset == 0) {
      // FIXME: this doesn't work for multiple clusters when idOffset may not be 0
      println("******WARNING****** gemmini tile id is 0! radiance tiles in the same cluster needs to be before gemmini")
    }
    val numPrevGemminis = prev.map(_.tileParams).map {
      case _: GemminiTileParams => 1
      case _ => 0
    }.sum
    val smKey = site(RadianceSharedMemKey).get
    val skipRecoding = false
    val tileParams = GemminiTileParams(
      gemminiConfig = {
        implicit val arithmetic: Arithmetic[Float] =
          Arithmetic.FloatArithmetic.asInstanceOf[Arithmetic[Float]]
        dataType match {
        case FP32 => GemminiFPConfigs.FP32DefaultConfig
        case FP16 => GemminiFPConfigs.FP16DefaultConfig.copy(
          acc_scale_args = Some(ScaleArguments(
            (t: Float, u: Float) => {t},
            1, Float(8, 24), -1, identity = "1.0", c_str = "((x))"
          )),
          mvin_scale_args = Some(ScaleArguments(
            (t: Float, u: Float) => t * u,
            1, Float(5, 11), -1, identity = "0x3c00", c_str="((x) * (scale))"
          )),
          mvin_scale_acc_args = None,
          has_training_convs = false,

          // from sirius
          spatialArrayInputType = Float(5, 11, isRecoded = skipRecoding),
          spatialArrayWeightType = Float(5, 11, isRecoded = skipRecoding),
          spatialArrayOutputType = Float(8, 24, isRecoded = skipRecoding),
          accType = Float(8, 24),
          // hardcode_d_to_garbage_addr = true,
          acc_read_full_width = false, // set to true to output fp32

          // acc_singleported = true,
          // clock_gate = true,
          num_counter = 0
        )
        case BF16 => GemminiFPConfigs.BF16DefaultConfig
        // TODO: Int8
      }}.copy(
        dataflow = Dataflow.WS,
        ex_read_from_acc = false,
        ex_write_to_spad = false,
        has_training_convs = false,
        has_max_pool = false,
        use_tl_ext_mem = true,
        sp_singleported = false,
        spad_read_delay = 4,
        use_shared_ext_mem = true,
        acc_sub_banks = 1,
        has_normalizations = false,
        meshRows = dim,
        meshColumns = dim,
        tile_latency = 0,
        mesh_output_delay = 1,
        acc_latency = 3,
        dma_maxbytes = site(CacheBlockBytes),
        dma_buswidth = dmaBytes,
        tl_ext_mem_base = smKey.address,
        sp_banks = smKey.numBanks,
        sp_capacity = CapacityInKilobytes(smKey.size >> 10),
        acc_capacity = CapacityInKilobytes(accSizeInKB),
      ),
      tileId = idOffset,
      tileSize = tileSize,
      slaveAddress = smKey.address + smKey.size + 0x3000 + 0x100 * numPrevGemminis
    )
    Seq(GemminiTileAttachParams(
      tileParams,
      crossing
    )) ++ prev
  }
  case NumTiles => up(NumTiles) + 1
}) {
  def this(location: HierarchicalLocation, dim: Int, accSizeInKB: Int, tileSize: Either[(Int, Int, Int), Int],
           dataType: RadianceGemminiDataType.Type = RadianceGemminiDataType.FP32, dmaBytes: Int = 256) =
    this(location, RocketCrossingParams(
      master = HierarchicalElementMasterPortParams.locationDefault(location),
      slave = HierarchicalElementSlavePortParams.locationDefault(location),
      mmioBaseAddressPrefixWhere = location match {
        case InSubsystem => CBUS
        case InCluster(clusterId) => CCBUS(clusterId)
      }
    ), dim, accSizeInKB, tileSize, dataType, dmaBytes)

  def this(location: HierarchicalLocation, dim: Int, accSizeInKB: Int, tileSize: Int) =
    this(location, dim, accSizeInKB, Right(tileSize))

  def this(location: HierarchicalLocation, dim: Int, accSizeInKB: Int, tileSize: (Int, Int, Int),
           dataType: RadianceGemminiDataType.Type) =
    this(location, dim, accSizeInKB, Left(tileSize), dataType)
}

class WithRadianceSharedMem(address: BigInt,
                            size: Int,
                            numBanks: Int,
                            numWords: Int,
                            memType: MemType = TwoPort,
                            strideByWord: Boolean = true,
                            filterAligned: Boolean = true,
                            disableMonitors: Boolean = true,
                            serializeUnaligned: RadianceSmemSerialization = FullySerialized
                           ) extends Config((_, _, _) => {
  case RadianceSharedMemKey => {
    require(isPow2(size) && size >= 1024)
    Some(RadianceSharedMemKey(
      address, size, numBanks, numWords, 4, memType,
      strideByWord, filterAligned, disableMonitors, serializeUnaligned
    ))
  }
})

class WithRadianceFrameBuffer(baseAddress: BigInt,
                              width: Int,
                              size: Int,
                              validAddress: BigInt,
                              fbName: String = "fb") extends Config((_, _, up) => {
  case RadianceFrameBufferKey => {
    up(RadianceFrameBufferKey) ++ Seq(
      RadianceFrameBufferKey(baseAddress, width, size, validAddress, fbName)
    )
  }
})

class WithRadianceCluster(
  clusterId: Int,
  location: HierarchicalLocation = InSubsystem,
  crossing: RocketCrossingParams = RocketCrossingParams()
) extends Config((site, here, up) => {
  case ClustersLocated(`location`) => up(ClustersLocated(location)) :+ RadianceClusterAttachParams(
    RadianceClusterParams(clusterId = clusterId),
    crossing)
  case TLNetworkTopologyLocated(InCluster(`clusterId`)) => List(
    ClusterBusTopologyParams(
      clusterId = clusterId,
      csbus = site(SystemBusKey),
      ccbus = site(ControlBusKey).copy(errorDevice = None),
      coherence = site(ClusterBankedCoherenceKey(clusterId))
    )
  )
  case PossibleTileLocations => up(PossibleTileLocations) :+ InCluster(clusterId)
})

// `nSrcIds`: number of source IDs for each mem lane.  This is for all warps
class WithSimtConfig(nWarps: Int = 4, nCoreLanes: Int = 4, nMemLanes: Int = 4, nSrcIds: Int = 8)
extends Config((site, _, up) => {
  case SIMTCoreKey => {
    Some(up(SIMTCoreKey).getOrElse(SIMTCoreParams()).copy(
      nWarps = nWarps,
      nCoreLanes = nCoreLanes,
      nMemLanes = nMemLanes,
      nSrcIds = nSrcIds
      ))
  }
})

class WithMemtraceCore(tracefilename: String, traceHasSource: Boolean = false)
extends Config((site, _, _) => {
  case MemtraceCoreKey => {
    require(
      site(SIMTCoreKey).isDefined,
      "Memtrace core requires a SIMT configuration. Use WithNLanes to enable SIMT."
    )
    Some(MemtraceCoreParams(tracefilename, traceHasSource))
  }
})

class WithPriorityCoalXbar extends Config((site, _, up) => {
  case CoalXbarKey => {
    Some(up(CoalXbarKey).getOrElse(CoalXbarParam))
  }
})

class WithVortexL1Banks(nBanks: Int = 4) extends Config ((site, here, up) => {
  case VortexL1Key => {
    Some(defaultVortexL1Config.copy(
      numBanks = nBanks,
      inputSize = up(SIMTCoreKey).get.nMemLanes * 4/*32b word*/,
      cacheLineSize = up(SIMTCoreKey).get.nMemLanes * 4/*32b word*/,
      memSideSourceIds = 16,
      mshrSize = 16,
    ))
  }
})

// When `enable` is false, we still elaborate Coalescer, but it acts as a
// pass-through logic that always outputs un-coalesced requests.  This is
// useful for when we want to keep the generated wire and net names the same
// to e.g. compare waveforms.
class WithCoalescer(nNewSrcIds: Int = 8, enable : Boolean = true) extends Config((site, _, up) => {
  case CoalescerKey => {
    val (nLanes, numOldSrcIds) = up(SIMTCoreKey) match {
      case Some(param) => (param.nMemLanes, param.nSrcIds)
      case None => (1,1)
    }

    val sbusWidthInBytes = site(SystemBusKey).beatBytes
    // FIXME: coalescer fails to instantiate with 4-byte bus
    require(sbusWidthInBytes > 2,
      "FIXME: coalescer currently doesn't instantiate with 4-byte sbus")

    // If instantiating L1 cache, the maximum coalescing size should match the
    // cache line size
    val maxCoalSizeInBytes = up(VortexL1Key) match {
      case Some(param) => param.inputSize
      case None => sbusWidthInBytes
    }
      
    // Note: this config chooses a single-sized coalescing logic by default.
    Some(DefaultCoalescerConfig.copy(
      enable       = enable,
      numLanes     = nLanes,
      numOldSrcIds = numOldSrcIds,
      numNewSrcIds = nNewSrcIds,
      addressWidth = 32, // FIXME hardcoded as 32-bit system
      dataBusWidth = log2Ceil(maxCoalSizeInBytes),
      coalLogSizes = Seq(log2Ceil(maxCoalSizeInBytes))
      )
    )
  }
})

class WithNCustomSmallRocketCores(
                             n: Int,
                             overrideIdOffset: Option[Int] = None,
                             crossing: RocketCrossingParams = RocketCrossingParams()
                           ) extends Config((site, here, up) => {
  case TilesLocated(InSubsystem) => {
    val prev = up(TilesLocated(InSubsystem))
    val idOffset = up(NumTiles)
    val med = RocketTileParams(
      core = RocketCoreParams(fpu = None),
      btb = None,
      dcache = Some(DCacheParams(
        rowBits = site(SystemBusKey).beatBits,
        nSets = 2,
        nWays = 1,
        nTLBSets = 1,
        nTLBWays = 2,
        nTLBBasePageSectors = 1,
        nTLBSuperpages = 1,
        nMSHRs = 0,
        blockBytes = site(CacheBlockBytes))),
      icache = Some(ICacheParams(
        rowBits = site(SystemBusKey).beatBits,
        nSets = 2,
        nWays = 1,
        nTLBSets = 1,
        nTLBWays = 2,
        nTLBBasePageSectors = 1,
        nTLBSuperpages = 1,
        blockBytes = site(CacheBlockBytes))))
    List.tabulate(n)(i => RocketTileAttachParams(
      med.copy(tileId = i + idOffset),
      crossing
    )) ++ prev
  }
  case NumTiles => up(NumTiles) + n
})

class WithExtGPUMem(address: BigInt = BigInt("0x100000000", 16),
                    size: BigInt = 0x80000000) extends Config((site, here, up) => {
  case GPUMemory() => Some(GPUMemParams(address, size))
  case ExtMem => up(ExtMem).map(x => {
    val gap = address - x.master.base - x.master.size
    x.copy(master = x.master.copy(size = x.master.size + gap + size))
  })
})
case class GPUMemParams(address: BigInt = BigInt("0x100000000", 16), size: BigInt = 0x80000000)
case class GPUMemory() extends Field[Option[GPUMemParams]](None)

object RadianceSimArgs extends Field[Option[Boolean]](None)

class WithRadianceSimParams(enabled: Boolean) extends Config((_, _, _) => {
  case RadianceSimArgs => Some(enabled)
})
