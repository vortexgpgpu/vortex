package freechips.rocketchip.tilelink.coalescing

import chisel3._
import chiseltest._
import chiseltest.simulator.VerilatorFlags
import org.scalatest.flatspec.AnyFlatSpec
import freechips.rocketchip.tilelink._
import freechips.rocketchip.util.MultiPortQueue
import freechips.rocketchip.diplomacy._
import freechips.rocketchip.subsystem.WithoutTLMonitors
import org.chipsalliance.cde.config.Parameters
import chisel3.util.{DecoupledIO, Valid}
import chisel3.util.experimental.BoringUtils


object coalArbTestConfig extends CoalescerConfig(
  enable = true,
  numLanes = 4,
  queueDepth = 1,
  waitTimeout = 8,
  addressWidth = 24,
  dataBusWidth = 5,
  // watermark = 2,
  wordSizeInBytes = 4,
  numOldSrcIds = 4,
  numNewSrcIds = 4,
  respQueueDepth = 4,
  coalLogSizes = Seq(4, 5),
  sizeEnum = DefaultInFlightTableSizeEnum,
  numCoalReqs = 1,
  numArbiterOutputPorts = 1,
  bankStrideInBytes = 64
)

class DummyCoalescerXbarUnitTB(implicit p: Parameters) extends LazyModule {
    val device = new SimpleDevice("dummy", Seq("dummy"))
    val beatBytes = 1 << coalArbTestConfig.dataBusWidth // 256 bit bus
    
    val l2Nodes = Seq.tabulate(coalArbTestConfig.numArbiterOutputPorts) { _ =>
        TLManagerNode(
        Seq(
            TLSlavePortParameters.v1(
            Seq(
                TLManagerParameters(
                address = Seq(AddressSet(0x8000000, 0xffffff)), // should be matching cpuNode
                resources = device.reg,
                regionType = RegionType.UNCACHED,
                executable = true,
                supportsArithmetic = TransferSizes(1, beatBytes),
                supportsLogical = TransferSizes(1, beatBytes),
                supportsGet = TransferSizes(1, beatBytes),
                supportsPutFull = TransferSizes(1, beatBytes),
                supportsPutPartial = TransferSizes(1, beatBytes),
                supportsHint = TransferSizes(1, beatBytes),
                fifoId = Some(0)
                )
            ),
            beatBytes
            )
        )
        )
    }

    val dut = LazyModule(new CoalescerXbar(coalArbTestConfig))

    l2Nodes.foreach(_ := dut.node)

    lazy val module = new DummyCoalescerXbarUnitTBImpl(this)
    
}
    class DummyCoalescerXbarUnitTBImpl(outer: DummyCoalescerXbarUnitTB) extends LazyModuleImp(outer) {

        val coalescerXbar = outer.dut

        val l2IOs       = Seq.tabulate(coalArbTestConfig.numArbiterOutputPorts){ i=>
            outer.l2Nodes(i).makeIOs()
        }

    }


class CoalescerXbarUnitTest extends AnyFlatSpec with ChiselScalatestTester {
    behavior of "testing various aspects of coalescer arbiter"

    implicit val p: Parameters = Parameters.empty

    it should "coalescer has not valid TL output" in {
    test(LazyModule(new DummyCoalescerXbarUnitTB()(new WithoutTLMonitors())).module)
    .withAnnotations(Seq(VerilatorBackendAnnotation, VerilatorFlags(Seq("--coverage-line")), WriteFstAnnotation))
    { c => 

        c.l2IOs.foreach(_.head.a.valid.expect(false.B))

    }
}

}
