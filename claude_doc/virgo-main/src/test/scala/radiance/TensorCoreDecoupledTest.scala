package radiance.core

import chisel3._
import chisel3.util._
import chiseltest._
import org.scalatest.flatspec.AnyFlatSpec

class TensorCoreDecoupledTest extends AnyFlatSpec with ChiselScalatestTester {
  behavior of "TensorCoreDecoupled"

  it should "do the right thing" in {
    test(new TensorCoreDecoupled(8, 8, numSourceIds = 4, half = true))
      { c =>
        c.io.initiate.valid.poke(true.B)
        c.io.initiate.bits.wid.poke(0.U)

        c.io.respA.valid.poke(false.B)
        c.io.respA.bits.data.poke(0.U)
        c.io.respB.valid.poke(false.B)
        c.io.respB.bits.data.poke(0.U)

        c.clock.step()
        c.io.writeback.valid.expect(true.B)
      }
  }
}
