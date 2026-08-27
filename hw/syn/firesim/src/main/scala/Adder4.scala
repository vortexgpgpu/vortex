//See LICENSE for license details.

// Toy target for the Vortex/U55C FireSim bring-up.
//
// Mirrors VerilogAccumulator, but wraps `adder4.v` -- a design whose port shape
// matches the contract Vortex's AFU must satisfy: a single clock input, a single
// reset, and no internally generated or gated clocks.

package firesim.midasexamples

import chisel3._
import chisel3.util.HasBlackBoxPath
import org.chipsalliance.cde.config.Parameters

class Adder4IO extends Bundle {
  val a   = Input(UInt(4.W))
  val b   = Input(UInt(4.W))
  val sum = Output(UInt(5.W))
}

/** BlackBox over the real `VX_adder4.v` in the Vortex tree.
  *
  * File-based rather than inline: this is the mechanism the Vortex AFU will use,
  * so the toy exercises it too. VORTEX_HOME overrides the default tree location.
  */
class Adder4Impl extends BlackBox with HasBlackBoxPath {
  // Chisel names the emitted module after the class unless told otherwise.
  // Pin it to `VX_adder4` so the instantiated module matches the module declared
  // in hw/rtl/afu/firesim/toy/VX_adder4.v.
  override def desiredName = "VX_adder4"

  val io = IO(new Bundle {
    val clk   = Input(Clock())
    val reset = Input(Bool())
    val a     = Input(UInt(4.W))
    val b     = Input(UInt(4.W))
    val sum   = Output(UInt(5.W))
  })
  // Point at the deliverable DUT in the Vortex tree. This is the same
  // file-based path the Vortex AFU blackbox will use. Required rather than
  // defaulted: a wrong tree would silently elaborate the wrong RTL.
  private val vortexHome = sys.env.getOrElse(
    "VORTEX_HOME",
    throw new RuntimeException("VORTEX_HOME must be set to the Vortex tree root"),
  )
  addPath(s"$vortexHome/hw/rtl/afu/firesim/toy/VX_adder4.v")
}

class Adder4DUT extends Module {
  val io   = IO(new Adder4IO)
  val impl = Module(new Adder4Impl)
  impl.io.clk   := clock
  impl.io.reset := reset.asBool
  impl.io.a     := io.a
  impl.io.b     := io.b
  io.sum        := impl.io.sum
}

class Adder4(implicit p: Parameters) extends firesim.lib.testutils.PeekPokeHarness(() => new Adder4DUT)
