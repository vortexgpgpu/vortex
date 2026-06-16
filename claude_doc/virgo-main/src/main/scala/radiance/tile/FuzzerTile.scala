// See LICENSE.SiFive for license details.
// See LICENSE.Berkeley for license details.

package radiance.tile

import chisel3._
import org.chipsalliance.cde.config.Parameters
import org.chipsalliance.diplomacy.lazymodule.LazyModule
import freechips.rocketchip.resources.SimpleDevice
import freechips.rocketchip.prci.ClockCrossingType
import freechips.rocketchip.rocket._
import freechips.rocketchip.tile._
import freechips.rocketchip.tilelink._
import freechips.rocketchip.subsystem.{HierarchicalElementCrossingParamsLike, CanAttachTile}
import freechips.rocketchip.prci.{ClockSinkParameters}
import radiance.core.{SIMTCoreKey}
import radiance.memory._

case class FuzzerTileParams(
    core: VortexCoreParams = VortexCoreParams(), // TODO: remove this
    useVxCache: Boolean = false,
    tileId: Int = 0,
) extends InstantiableTileParams[FuzzerTile] {
  def instantiate(crossing: HierarchicalElementCrossingParamsLike, lookup: LookupByHartIdImpl)(
      implicit p: Parameters
  ): FuzzerTile = {
    new FuzzerTile(this, crossing, lookup)
  }
  val clockSinkParams = ClockSinkParameters()
  val blockerCtrlAddr = None
  val icache = None
  val dcache = None
  val btb = None
  val baseName = "radiance_fuzzer_tile"
  val uniqueName = s"${baseName}_$tileId"
}

case class FuzzerTileAttachParams(
  tileParams: FuzzerTileParams,
  crossingParams: HierarchicalElementCrossingParamsLike
) extends CanAttachTile { type TileType = FuzzerTile }

class FuzzerTile private (
    val fuzzerParams: FuzzerTileParams,
    crossing: ClockCrossingType,
    lookup: LookupByHartIdImpl,
    q: Parameters
) extends BaseTile(fuzzerParams, crossing, lookup, q)
    with SinksExternalInterrupts
    with SourcesExternalNotifications {
  def this(
      params: FuzzerTileParams,
      crossing: HierarchicalElementCrossingParamsLike,
      lookup: LookupByHartIdImpl
  )(implicit p: Parameters) =
    this(params, crossing.crossingType, lookup, p)

  val cpuDevice: SimpleDevice = new SimpleDevice("fuzzer", Nil)

  val intOutwardNode = None
  val slaveNode: TLInwardNode = TLIdentityNode()
  val masterNode = visibilityNode
  // val statusNode = BundleBridgeSource(() => new GroundTestStatus)

  val (numLanes, numSrcIds) = p(SIMTCoreKey) match {
      case Some(param) => (param.nMemLanes, param.nSrcIds)
      case None => {
        require(false, "fuzzer requires SIMTCoreKey to be defined")
        (0, 0)
      }
  }
  // FIXME: parameterize
  val wordSizeInBytes = 4

  val fuzzer = LazyModule(new MemFuzzer(numLanes, numSrcIds, wordSizeInBytes))

  // Conditionally instantiate memory coalescer
  val coalescerNode = p(CoalescerKey) match {
    case Some(coalParam) => {
      val coal = LazyModule(new CoalescingUnit(coalParam))
      coal.cpuNode :=* TLWidthWidget(4) :=* fuzzer.node
      coal.aggregateNode
    }
    case None => fuzzer.node
  }

  masterNode :=* coalescerNode

  override lazy val module = new FuzzerTileModuleImp(this)
}

class FuzzerTileModuleImp(outer: FuzzerTile) extends BaseTileModuleImp(outer) {
  outer.reportCease(Some(outer.fuzzer.module.io.finished))
}
