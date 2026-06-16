package radiance.memory

import freechips.rocketchip.subsystem._
import org.chipsalliance.cde.config.Parameters

trait CanHaveRadianceROMs { this: BaseSubsystem =>
  implicit val p: Parameters
  p(RadianceROMsLocated()).foreach(_.foreach { rom => RadianceROM.attachROM(rom, this, CBUS) })
}
