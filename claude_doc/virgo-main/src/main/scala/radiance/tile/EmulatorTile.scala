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
import radiance.core._
import radiance.memory.{CoalescingUnit, CoalescerKey}

// TODO: De-duplicate between this and FuzzerTile

case class EmulatorTileParams(
    core: VortexCoreParams = VortexCoreParams(), // TODO: remove this
    useVxCache: Boolean = false,
    tileId: Int = 0,
) extends InstantiableTileParams[EmulatorTile] {
  def instantiate(crossing: HierarchicalElementCrossingParamsLike, lookup: LookupByHartIdImpl)(
      implicit p: Parameters
  ): EmulatorTile = {
    new EmulatorTile(this, crossing, lookup)
  }
  val clockSinkParams = ClockSinkParameters()
  val blockerCtrlAddr = None
  val icache = None
  val dcache = None
  val btb = None
  val baseName = "radiance_emulator_tile"
  val uniqueName = s"${baseName}_$tileId"
}

case class EmulatorTileAttachParams(
  tileParams: EmulatorTileParams,
  crossingParams: HierarchicalElementCrossingParamsLike
) extends CanAttachTile { type TileType = EmulatorTile }

class EmulatorTile private (
    val EmulatorParams: EmulatorTileParams,
    crossing: ClockCrossingType,
    lookup: LookupByHartIdImpl,
    q: Parameters
) extends BaseTile(EmulatorParams, crossing, lookup, q)
    with SinksExternalInterrupts
    with SourcesExternalNotifications {
  def this(
      params: EmulatorTileParams,
      crossing: HierarchicalElementCrossingParamsLike,
      lookup: LookupByHartIdImpl
  )(implicit p: Parameters) =
    this(params, crossing.crossingType, lookup, p)

  val cpuDevice: SimpleDevice = new SimpleDevice("emulator", Nil)

  val intOutwardNode = None
  val slaveNode: TLInwardNode = TLIdentityNode()
  val masterNode = visibilityNode
  // val statusNode = BundleBridgeSource(() => new GroundTestStatus)

  val (numLanes, numSrcIds) = p(SIMTCoreKey) match {
      case Some(param) => (param.nMemLanes, param.nSrcIds)
      case None => {
        require(false, "emulator requires SIMTCoreKey to be defined")
        (0, 0)
      }
  }
  // FIXME: parameterize
  val wordSizeInBytes = 4

  val emulator = LazyModule(new Emulator(numLanes, numSrcIds, wordSizeInBytes))

  // Conditionally instantiate memory coalescer
  val coalescerNode = p(CoalescerKey) match {
    case Some(coalParam) => {
      val coal = LazyModule(new CoalescingUnit(coalParam))
      coal.cpuNode :=* TLWidthWidget(4) :=* emulator.node
      coal.aggregateNode
    }
    case None => emulator.node
  }

  masterNode :=* coalescerNode

  override lazy val module = new EmulatorTileModuleImp(this)
}

class EmulatorTileModuleImp(outer: EmulatorTile) extends BaseTileModuleImp(outer) {
  outer.reportCease(Some(outer.emulator.module.io.finished))
}
