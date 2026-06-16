// See LICENSE.SiFive for license details.

package radiance.unittest

import chisel3._
import org.chipsalliance.cde.config._
import freechips.rocketchip.subsystem.{BaseSubsystemConfig}
import freechips.rocketchip.devices.tilelink._
import freechips.rocketchip.tilelink._
import freechips.rocketchip.util._
import radiance.core.TensorCoreDecoupledTest
import radiance.memory._
import radiance.subsystem.WithSimtConfig
import freechips.rocketchip.unittest._
//import rocket.VortexFatBankTest

case object TestDurationMultiplier extends Field[Int]

class WithTestDuration(x: Int) extends Config((site, here, up) => {
  case TestDurationMultiplier => x
})

class WithTensorUnitTests extends Config((site, _, _) => {
  case UnitTests => (q: Parameters) => {
    implicit val p = q
    val timeout = 50000 * site(TestDurationMultiplier)
    Seq(
      Module(new TensorCoreDecoupledTest(timeout=timeout)),
    ) }
})

class WithCoalescingUnitTests extends Config((site, _, _) => {
  case UnitTests => (q: Parameters) => {
    implicit val p = q
    val timeout = 50000 * site(TestDurationMultiplier)
    Seq(
      // Module(new TLRAMCoalescerTest(timeout=timeout)),
      Module(new TLRAMCoalescerLoggerTest(filename="vecadd.core1.thread4.trace", timeout=timeout)),
      // Module(new TLRAMCoalescerLoggerTest(filename="sfilter.core1.thread4.trace", timeout=timeout)),
      // Module(new TLRAMCoalescerLoggerTest(filename="nearn.core1.thread4.trace", timeout=50000000 * site(TestDurationMultiplier))),
      // Module(new TLRAMCoalescerLoggerTest(filename="psort.core1.thread4.trace", timeout=timeout)),
      // Module(new TLRAMCoalescerLoggerTest(filename="nvbit.vecadd.n100000.filter_sm0.trace", timeout=timeout)(new WithSimtConfig(32))),
      // Module(new TLRAMCoalescerLoggerTest(filename="nvbit.vecadd.n100000.filter_sm0.lane4.trace", timeout=timeout)),
    ) }
})

/*
class WithVortexFatBankUnitTests extends Config((site, _, _) => {
  case UnitTests => (q: Parameters) => {
    implicit val p = q
    val timeout = 50000 * site(TestDurationMultiplier)
    Seq(
      Module(new VortexFatBankTest(filename="oclprintf.core1.thread4.trace", timeout=timeout)),
    )}
})
*/

class WithCoalescingUnitSynthesisDummy(nLanes: Int) extends Config((site, _, _) => {
  case UnitTests => (q: Parameters) => {
    implicit val p = q
    val timeout = 50000 * site(TestDurationMultiplier)
    Seq(
      Module(new DummyCoalescerTest(timeout=timeout)(new WithSimtConfig(nMemLanes=4))),
    ) }
})

class TensorUnitTestConfig extends Config(
  new WithTensorUnitTests ++
  new WithTestDuration(10) ++
  new BaseSubsystemConfig)

class CoalescingUnitTestConfig extends Config(
  new WithCoalescingUnitTests ++
  new WithTestDuration(10) ++
  new WithSimtConfig(nMemLanes=4) ++
  new BaseSubsystemConfig)

//class VortexFatBankUnitTestConfig extends Config(new WithVortexFatBankUnitTests ++ new WithTestDuration(10) ++ new WithSimtConfig(nLanes=4) ++ new BaseSubsystemConfig)

// Dummy configs of various sizes for synthesis
class CoalescingSynthesisDummyLane4Config extends Config(
  new WithCoalescingUnitSynthesisDummy(4) ++
  new WithTestDuration(10) ++
  new BaseSubsystemConfig)
class CoalescingSynthesisDummyLane8Config extends Config(
  new WithCoalescingUnitSynthesisDummy(8) ++
  new WithTestDuration(10) ++
  new BaseSubsystemConfig)
class CoalescingSynthesisDummyLane16Config extends Config(
  new WithCoalescingUnitSynthesisDummy(16) ++
  new WithTestDuration(10) ++
  new BaseSubsystemConfig)
class CoalescingSynthesisDummyLane32Config extends Config(
  new WithCoalescingUnitSynthesisDummy(32) ++
  new WithTestDuration(10) ++
  new BaseSubsystemConfig)

