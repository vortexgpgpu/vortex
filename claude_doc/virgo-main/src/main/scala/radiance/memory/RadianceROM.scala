// See LICENSE.SiFive for license details.

package radiance.memory

import chisel3._
import chisel3.util.log2Ceil
import org.chipsalliance.cde.config.{Config, Field, Parameters}
import freechips.rocketchip.subsystem.{BaseSubsystem, HierarchicalLocation, InSubsystem, TLBusWrapperLocation}
import freechips.rocketchip.diplomacy._
import freechips.rocketchip.tilelink._
import freechips.rocketchip.devices.tilelink._
import freechips.rocketchip.prci.ClockSinkDomain

import java.nio.ByteBuffer
import java.nio.file.{Files, Paths}

/** Size, location and contents of the boot rom. */
case class RadianceROMParams(address: BigInt,
                                 size: Int = 0x10000,
                                 contentFileName: String)
case class RadianceROMsLocated() extends Field[Option[Seq[RadianceROMParams]]](None)

object RadianceROM {
  /** BootROM.attach not only instantiates a TLROM and attaches it to the tilelink interconnect
    *    at a configurable location, but also drives the tiles' reset vectors to point
    *    at its 'hang' address parameter value.
    */
  def attach(params: BootROMParams, subsystem: BaseSubsystem, where: TLBusWrapperLocation,
             driveResetVector: Boolean = true) (implicit p: Parameters): TLROM = {
    val tlbus = subsystem.locateTLBusWrapper(where)
    val bootROMDomainWrapper = LazyModule(new ClockSinkDomain(take = None))
    bootROMDomainWrapper.clockNode := tlbus.fixedClockNode

    lazy val contents = {
      val romdata = Files.readAllBytes(Paths.get(params.contentFileName))
      val rom = ByteBuffer.wrap(romdata)
      rom.array() ++ subsystem.dtb.contents
    }

    val bootrom = bootROMDomainWrapper {
      LazyModule(new TLROM(params.address, params.size, contents, true, tlbus.beatBytes))
    }

    bootrom.node := tlbus.coupleTo("bootrom"){ TLFragmenter(tlbus) := _ }

    bootrom
  }

  def attachROM(params: RadianceROMParams, subsystem: BaseSubsystem, where: TLBusWrapperLocation)
                (implicit p: Parameters): Unit = {
    attach(BootROMParams(address = params.address, size = params.size, contentFileName = params.contentFileName),
      subsystem, where, driveResetVector = false)
  }
}
