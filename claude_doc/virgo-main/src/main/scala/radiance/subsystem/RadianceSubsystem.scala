// See LICENSE.SiFive for license details.

package radiance.subsystem

import freechips.rocketchip.subsystem._
import radiance.tile._

case class RadianceTileAttachParams(
  tileParams: RadianceTileParams,
  crossingParams: RocketCrossingParams
) extends CanAttachTile { type TileType = RadianceTile }

case class RadianceClusterAttachParams (
  clusterParams: RadianceClusterParams,
  crossingParams: HierarchicalElementCrossingParamsLike
) extends CanAttachCluster {
  type ClusterType = RadianceCluster
}
