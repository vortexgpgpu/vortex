package freechips.rocketchip.tilelink.coalescing

import chisel3._
import chisel3.stage.PrintFullStackTraceAnnotation
import chiseltest._
import chiseltest.simulator.VerilatorFlags
import org.scalatest.flatspec.AnyFlatSpec
import freechips.rocketchip.tilelink._
import freechips.rocketchip.util.MultiPortQueue
import freechips.rocketchip.diplomacy._
import freechips.rocketchip.subsystem.WithoutTLMonitors
import org.chipsalliance.cde.config.Parameters
import chisel3.util.{Cat, DecoupledIO, Valid}
import chisel3.util.experimental.BoringUtils

class MultiPortQueueUnitTest extends AnyFlatSpec with ChiselScalatestTester {
  behavior of "MultiPortQueue"

  // This is really just to figure out how MultiPortQueue works
  it should "serialize at dequeue end" in {
    test(new MultiPortQueue(UInt(4.W), 3, 1, 3, 6))
      .withAnnotations(Seq(WriteVcdAnnotation)) { c =>
        c.io.enq(0).valid.poke(true.B)
        c.io.enq(0).bits.poke(11.U)
        c.io.enq(1).valid.poke(true.B)
        c.io.enq(1).bits.poke(15.U)
        c.io.enq(2).valid.poke(true.B)
        c.io.enq(2).bits.poke(7.U)
        c.io.deq(0).ready.poke(true.B)
        c.clock.step()
        // c.io.enq(0).valid.poke(false.B)
        // c.io.enq(1).valid.poke(false.B)
        for (_ <- 0 until 100) {
          c.clock.step()
        }
      // c.io.deq(0).valid.expect(false.B)
      }
  }
}

class Splitter(implicit p: Parameters) extends LazyModule {
//  private def noChangeRequired(manager: TLManagerPortParameters) = manager.beatBytes == 4
  val node = new TLNexusNode(
    clientFn = { c =>
      require(c.length == 1, "splitter client check")
      require(c.head.clients.length == 1, "splitter client check")
      val headId = c.head.clients.head.sourceId
      c.head.v1copy(
        clients = Seq.tabulate(4)(i => c.head.clients.head.v1copy(
          sourceId = headId.shift(headId.size * i),
          supportsProbe = TransferSizes(1, 8),
          supportsArithmetic = TransferSizes(1, 8),
          supportsLogical = TransferSizes(1, 8),
          supportsGet = TransferSizes(1, 8),
          supportsPutFull = TransferSizes(1, 8),
          supportsPutPartial = TransferSizes(1, 8)
        ))
      )
    },
    managerFn = { m =>
      require(m.length == 4, "splitter manager check")
//      require(m.head.managers.length == 4, "splitter manager check")
      m.head.v1copy(
        beatBytes = 32,
        managers = Seq.fill(4)(m.head.managers.head.v1copy(
          address = Seq(AddressSet(0, 0xffffff)), // full range
          supportsGet = TransferSizes(1, 32),
          supportsPutFull = TransferSizes(1, 32),
          supportsPutPartial = TransferSizes(1, 32),
        ))
      )
    }
  ) //{
//    override def circuitIdentity = edges.out.map(_.manager).forall(noChangeRequired)
//  }
  lazy val module = new SplitterImp(this)
}

class SplitterImp(outer: Splitter) extends LazyModuleImp(outer) {
  val node = outer.node
  require(node.in.length == 1, "there should only be one edge in")
  require(node.in.head._2.manager.beatBytes == 32, "edge in should be 256 bits")
  require(node.out.length == 4, "there should be 4 edges out")
  require(node.out.head._2.manager.beatBytes == 8, "edge out should be 64 bits")

  val (in, edgeIn) = node.in.head

  in.a.ready := node.out.map(_._1.a.ready).reduce(_ && _)
  in.d.valid := node.out.map(_._1.d.valid).reduce(_ && _)
  in.d.bits.data := Cat(node.out.map(_._1.d.bits.data).reverse)
  in.d.bits.size := 5.U // FIXME: this is often wrong

  node.out.zipWithIndex.foreach { case ((out, edgeOut), i) =>
    when (!in.a.valid || in.a.bits.size === 5.U) {
      printf("[WARNING] runtime request size is not 256 bits")
    }
    assert(!out.d.valid || out.d.bits.size === 3.U, "runtime response size is not 64 bits")

    out.a.valid := in.a.valid
    out.a.bits := in.a.bits
    out.a.bits.size := in.a.bits.size.min(3.U)
    out.a.bits.address := in.a.bits.address | (i << 3).U
    out.a.bits.data := in.a.bits.data(64 * (i + 1) - 1, 64 * i)
    out.a.bits.mask := in.a.bits.mask(8 * (i + 1) - 1, 8 * i)

    out.d.ready := in.d.ready && in.d.valid // this might not conform to deadlock rules
  }
}

class DummyCoalescingUnitTB(implicit p: Parameters) extends LazyModule {
  val cpuNodes = Seq.tabulate(testConfig.numLanes) { _ =>
    TLClientNode(
      Seq(
        TLMasterPortParameters.v1(
          Seq(
            TLClientParameters(
              name = "processor-nodes",
              sourceId = IdRange(0, testConfig.numOldSrcIds),
              visibility = Seq(AddressSet(0x0, 0xffffff))
            )
          )
        )
      )
    ) // 24 bit address space (TODO probably use testConfig)
  }

  val device = new SimpleDevice("dummy", Seq("dummy"))
  val beatBytes = 1 << testConfig.dataBusWidth // 256 bit bus
  val numBanks = 4
  val bankWidth = beatBytes / numBanks // 8 bytes
  val l2Nodes = Seq.tabulate(4) { bank =>
    TLManagerNode(
      Seq(
        TLSlavePortParameters.v1(
          Seq(
            TLManagerParameters(
              address = Seq(AddressSet(bank * bankWidth, 0xffffff ^ ((numBanks - 1) * bankWidth))),
              resources = device.reg,
              regionType = RegionType.UNCACHED,
              executable = true,
              supportsArithmetic = TransferSizes(1, beatBytes min bankWidth),
              supportsLogical = TransferSizes(1, beatBytes min bankWidth),
              supportsGet = TransferSizes(1, beatBytes min bankWidth),
              supportsPutFull = TransferSizes(1, beatBytes min bankWidth),
              supportsPutPartial = TransferSizes(1, beatBytes min bankWidth),
              supportsHint = TransferSizes(1, beatBytes min bankWidth),
              fifoId = Some(0)
            )
          ),
          beatBytes min bankWidth
        )
      )
    )
  }

  val dut = LazyModule(new CoalescingUnit(testConfig))

  cpuNodes.foreach(dut.cpuNode := _)


  val splitters = Seq.fill(5)(LazyModule(new Splitter()))

  val splitOuts = Seq.fill(5)(Seq.fill(4)(TLIdentityNode()))


//  val xbar = TLXbar()
//  (splitters zip splitOuts).foreach { case (splitter, splitOut) =>
//    splitter.node := dut.aggregateNode
//    splitOut.foreach(xbar := _ := splitter.node)
//  }
//  l2Nodes.foreach(_ := xbar)


  (splitters zip splitOuts).foreach { case (splitter, splitOut) =>
    splitter.node := dut.aggregateNode
    splitOut.foreach(_ := splitter.node)
  }

  val xbars = Seq.fill(4)(TLXbar()) // per bank xbar that arbitrates between N+1
  splitOuts.foreach { allBanks =>
    (allBanks zip xbars).foreach { case (splitBank, xbar) =>
      xbar := splitBank
    }
  }
  (l2Nodes zip xbars).foreach { case (l2Node, xbar) =>
    l2Node := xbar
  }


  lazy val module = new DummyCoalescingUnitTBImp(this)
}

class DummyCoalescingUnitTBImp(outer: DummyCoalescingUnitTB) extends LazyModuleImp(outer) {
//  println(s"aggregate node max transfer size ${outer.dut.aggregateNode.out.head._2.maxTransfer}")
//  println(s"splitter in max transfer size ${outer.splitters.head.node.in.head._2.maxTransfer}")
//  println(s"splitter out max transfer size ${outer.splitters.head.node.out.head._2.maxTransfer}")

  val coal = outer.dut

//  (outer.splitters.map(_.node.in.head) zip coal.aggregateNode.out).foreach { case ((in, inEdge), (out, outEdge)) =>
//
//  }

  // FIXME: these need to be separate variables because of implicit naming in makeIOs
  // there has to be a better way
  val coalIO0 = outer.cpuNodes(0).makeIOs()
  val coalIO1 = outer.cpuNodes(1).makeIOs()
  val coalIO2 = outer.cpuNodes(2).makeIOs()
  val coalIO3 = outer.cpuNodes(3).makeIOs()
  val coalIOs = Seq(coalIO0, coalIO1, coalIO2, coalIO3)

  val l2IO0 = outer.l2Nodes(0).makeIOs()
  val l2IO1 = outer.l2Nodes(1).makeIOs()
  val l2IO2 = outer.l2Nodes(2).makeIOs()
  val l2IO3 = outer.l2Nodes(3).makeIOs()
  val l2IOs = Seq(l2IO0, l2IO1, l2IO2, l2IO3)

//  val coalMasterNode = coal.coalescerNode.makeIOs()

  private val reqQueues = coal.module.reqQueues
  private val coalescer = coal.module.coalescer

  // workaround for peeking internal signals as outlined in
  // https://github.com/ucb-bar/chiseltest/issues/17

  private val peekIn = Seq(
    reqQueues.io.queue.enq.map(_.ready),
    reqQueues.io.queue.enq.map(_.bits),
    reqQueues.io.queue.enq.map(_.valid),
    reqQueues.io.queue.deq.map(_.bits),
    reqQueues.io.queue.deq.map(_.valid),
    coalescer.io.coalReq.ready,
    coalescer.io.coalReq.bits,
    coalescer.io.coalReq.valid,
    coalescer.io.invalidate,
  )

  val reqQueueEnqReady =  peekIn(0).asInstanceOf[Seq[Bool]].map(x => IO(x.cloneType))
  val reqQueueEnqBits =   peekIn(1).asInstanceOf[Seq[Request]].map(x => IO(x.cloneType))
  val reqQueueEnqValid =  peekIn(2).asInstanceOf[Seq[Bool]].map(x => IO(x.cloneType))
  val reqQueueDeqBits =   peekIn(3).asInstanceOf[Seq[Request]].map(x => IO(Output(x.cloneType)))
  val reqQueueDeqValid =  peekIn(4).asInstanceOf[Seq[Bool]].map(x => IO(Output(x.cloneType)))
  val coalReqReady =      IO(Output(peekIn(5).asInstanceOf[Bool].cloneType))
  val coalReqBits =       IO(Output(peekIn(6).asInstanceOf[Request].cloneType))
  val coalReqValid =      IO(Output(peekIn(7).asInstanceOf[Bool].cloneType))
  val coalInvalidate =    IO(Output(peekIn(8).asInstanceOf[Valid[Vec[UInt]]].cloneType))

  private val peekOut = Seq(
    reqQueueEnqReady, reqQueueEnqBits, reqQueueEnqValid,
    reqQueueDeqBits, reqQueueDeqValid,
    coalReqReady, coalReqBits, coalReqValid, coalInvalidate,
  )

  (peekIn zip peekOut).foreach {
    case (inner: IndexedSeq[Data], outer: Seq[Data]) =>
      (inner zip outer).foreach { case (i, o) =>
        BoringUtils.bore(i, Seq(o))
      }
    case (inner: Data, outer: Data) =>
      BoringUtils.bore(inner, Seq(outer))
    case _ =>
      assert(false, "boring between different data types")
  }

  private val pokeIn = Seq(
    reqQueues.io.queue.deq.map(_.ready),
//    coalescer.io.coalReq.ready
  )

  val reqQueueDeqReady = pokeIn(0).asInstanceOf[Seq[Bool]].map(x => IO(x.cloneType))

  private val pokeOut = Seq(
    reqQueueDeqReady
  )

  // TODO: doesn't work yet
  (pokeIn zip pokeOut).foreach {
    case (inner: IndexedSeq[Data], outer: Seq[Data]) =>
      (inner zip outer).foreach { case (i, o) =>
        BoringUtils.bore(i, Seq(o))
      }
    case (inner: Data, outer: Data) =>
      BoringUtils.bore(inner, Seq(outer))
    case _ =>
      assert(false, "boring between different data types")
  }
}

object testConfig extends CoalescerConfig(
  enable = true,
  numLanes = 4,
  queueDepth = 1,
  waitTimeout = 8,
  addressWidth = 24,
  dataBusWidth = 5,
  // watermark = 2,
  wordSizeInBytes = 4,
  numOldSrcIds = 16,
  numNewSrcIds = 4,
  respQueueDepth = 4,
  coalLogSizes = Seq(4, 5),
  sizeEnum = DefaultInFlightTableSizeEnum,
  numCoalReqs = 1,
  numArbiterOutputPorts = 4,
  bankStrideInBytes = 64
)

class CoalescerUnitTest extends AnyFlatSpec with ChiselScalatestTester {
  behavior of "multi- and mono-coalescers"

  implicit val p: Parameters = Parameters.empty

  def pokeA(
      nodes: Seq[TLBundle],
      idx: Int, op: Int, size: Int, source: Int, addr: Int, mask: Int, data: Long,
      valid: Boolean = true,
  ): Unit = {
    val node = nodes(idx)
    node.a.ready.expect(true.B)
    node.a.bits.opcode.poke(if (op == 1) TLMessages.PutFullData else TLMessages.Get)
    node.a.bits.param.poke(0.U)
    node.a.bits.size.poke(size.U)
    node.a.bits.source.poke(source.U)
    node.a.bits.address.poke(addr.U)
    node.a.bits.mask.poke(mask.U)
    node.a.bits.data.poke(data.U)
    node.a.bits.corrupt.poke(false.B)
    node.a.valid.poke(valid.B)
  }

  def unsetA(nodes: Seq[TLBundle]): Unit = {
    nodes.foreach { node =>
      node.a.valid.poke(false.B)
    }
  }

  def expectVec[T <: Data](vec: Seq[T], value: Seq[T]): Unit = {
    (vec zip value).foreach { case (a, b) => a.expect(b) }
  }

  it should "coalesce fully consecutive accesses at size 4, only once" in {
    test(LazyModule(new DummyCoalescingUnitTB()(new WithoutTLMonitors())).module)
    .withAnnotations(Seq(VerilatorBackendAnnotation, VerilatorFlags(Seq("--coverage-line")), WriteFstAnnotation, PrintFullStackTraceAnnotation))
//    .withAnnotations(Seq(VcsBackendAnnotation, WriteFsdbAnnotation))
    { c =>
      val nodes = c.coalIOs.map(_.head)

      c.l2IOs.foreach(_.head.a.ready.poke(true.B))

      c.reqQueueEnqReady.foreach(_.expect(true.B))
      pokeA(nodes, idx = 0, op = 1, size = 2, source = 0, addr = 0x10, mask = 0xf, data = 0x1111)
      pokeA(nodes, idx = 1, op = 1, size = 2, source = 0, addr = 0x14, mask = 0xf, data = 0x2222)
      pokeA(nodes, idx = 2, op = 1, size = 2, source = 0, addr = 0x18, mask = 0xf, data = 0x3333)
      pokeA(nodes, idx = 3, op = 1, size = 2, source = 0, addr = 0x1c, mask = 0xf, data = 0x4444)
      expectVec(c.reqQueueEnqBits.map(_.data), Seq(0x1111.U, 0x2222.U, 0x3333.U, 0x4444.U))
      c.clock.step()

      unsetA(nodes)
      c.reqQueueDeqValid.foreach(_.expect(false.B))

      c.coalReqValid.expect(true.B)
      c.coalReqBits.address.expect(0x10.U)
      c.coalReqBits.data.expect(BigInt("4444000033330000222200001111", 16) << 128)
      c.coalReqBits.mask.expect(0xffff0000L)
      c.coalReqBits.size.expect(4.U)
      c.coalReqBits.op.expect(1.U)

      c.coalReqReady.expect(true.B)
      c.reqQueueEnqReady.foreach(_.expect(true.B))
      pokeA(nodes, idx = 0, op = 1, size = 2, source = 1, addr = 0xf20, mask = 0xf, data = 0x5555)
      pokeA(nodes, idx = 1, op = 1, size = 2, source = 1, addr = 0xf24, mask = 0xf, data = 0x6666, valid = false)
      pokeA(nodes, idx = 2, op = 1, size = 2, source = 1, addr = 0xf28, mask = 0xf, data = 0x7777)
      pokeA(nodes, idx = 3, op = 1, size = 2, source = 1, addr = 0xf2c, mask = 0xf, data = 0x8888, valid = false)
      c.clock.step()

      c.coalReqValid.expect(true.B)
      c.coalReqBits.address.expect(0xf20.U)
      c.coalReqBits.data.expect(BigInt("77770000000000005555", 16)) // technically these can be dontcare's
      c.coalReqBits.mask.expect(0x00000f0f)
      c.coalReqBits.size.expect(4.U)
      c.coalReqBits.op.expect(1.U)

      c.coalReqReady.expect(true.B)
      c.reqQueueEnqReady.foreach(_.expect(true.B))
      pokeA(nodes, idx = 0, op = 0, size = 2, source = 2, addr = 0xd04, mask = 0xa, data = 0xdeadbeefL)
      pokeA(nodes, idx = 1, op = 0, size = 2, source = 2, addr = 0xd0c, mask = 0xb, data = 0x8badf00dL)
      pokeA(nodes, idx = 2, op = 0, size = 2, source = 2, addr = 0xd14, mask = 0xc, data = 0xcafeb0baL)
      pokeA(nodes, idx = 3, op = 0, size = 2, source = 2, addr = 0xd1c, mask = 0xd, data = 0xdabbad00L)
      c.clock.step()

      c.coalReqValid.expect(true.B)
      c.coalReqBits.address.expect(0xd00.U)
      c.coalReqBits.size.expect(5.U)
      c.coalReqBits.data.expect(BigInt("dabbad0000000000cafeb0ba000000008badf00d00000000deadbeef00000000", 16))
      c.coalReqBits.mask.expect(0xd0c0b0a0L)
      c.coalReqBits.op.expect(0.U)

      c.clock.step()
//      c.clock.step()
    }
  }

  it should "coalesce identical addresses (stride of 0)" in {
    test(LazyModule(new DummyCoalescingUnitTB()(new WithoutTLMonitors())).module)
    .withAnnotations(Seq(VerilatorBackendAnnotation))
    { c =>
      println(s"coalIO length = ${c.coalIOs(0).length}")
      val nodes = c.coalIOs.map(_.head)

      c.l2IOs.foreach(_.head.a.ready.poke(true.B))
      c.coalReqReady.expect(true.B)
      c.reqQueueEnqReady.foreach(_.expect(true.B))
      pokeA(nodes, idx = 0, op = 1, size = 2, source = 0, addr = 0x18, mask = 0xf, data = 0x1111)
      pokeA(nodes, idx = 1, op = 1, size = 2, source = 0, addr = 0x18, mask = 0xf, data = 0x2222)
      pokeA(nodes, idx = 2, op = 1, size = 2, source = 0, addr = 0x18, mask = 0xf, data = 0x3333)
      pokeA(nodes, idx = 3, op = 1, size = 2, source = 0, addr = 0x18, mask = 0xf, data = 0x4444)

      c.clock.step()

      unsetA(nodes)
      c.coalReqValid.expect(true.B)
      c.coalReqBits.address.expect(0x10.U)
      c.coalReqBits.data.expect(BigInt("11110000000000000000", 16) << 128) // could be any of the 4 reqs
      c.coalReqBits.mask.expect(0x0f000000)
      c.coalReqBits.size.expect(4.U)
      c.coalReqBits.op.expect(1.U)

      c.clock.step()
    }
  }

  it should "coalesce the coalescable chunk and leave 2 uncoalescable requests" in {
    test(LazyModule(new DummyCoalescingUnitTB()).module)
//      .withAnnotations(Seq(VcsBackendAnnotation)) { c =>
      .withAnnotations(Seq(VerilatorBackendAnnotation)) { c =>
        println(s"coalIO length = ${c.coalIOs(0).length}")
        val nodes = c.coalIOs.map(_.head)

        c.l2IOs.foreach(_.head.a.ready.poke(true.B))
        c.coalReqReady.expect(true.B)
        c.reqQueueEnqReady.foreach(_.expect(true.B))
        pokeA(nodes, idx = 0, op = 1, size = 2, source = 0, addr = 0x04, mask = 0xf, data = 0x1111)
        pokeA(nodes, idx = 1, op = 1, size = 2, source = 0, addr = 0x08, mask = 0xf, data = 0x2222)
        pokeA(nodes, idx = 2, op = 1, size = 2, source = 0, addr = 0xf00, mask = 0xf, data = 0x3333)
        pokeA(nodes, idx = 3, op = 1, size = 2, source = 0, addr = 0xd00, mask = 0xf, data = 0x4444)

        c.clock.step()

        unsetA(nodes)
        c.coalReqValid.expect(true.B)
        c.coalReqBits.address.expect(0x00.U)
        c.coalReqBits.data.expect(BigInt("22220000111100000000", 16))
        c.coalReqBits.mask.expect(0x00000ff0)
        c.coalReqBits.size.expect(4.U)
        c.coalReqBits.op.expect(1.U)

        expectVec(c.reqQueueDeqValid, Seq(false.B, false.B, true.B, true.B))
        expectVec(c.reqQueueDeqBits.map(_.data), Seq(0x1111.U, 0x2222.U, 0x3333.U, 0x4444.U))
        expectVec(c.reqQueueDeqReady, Seq.fill(4)(true.B))

        c.clock.step()
        expectVec(c.reqQueueDeqValid, Seq.fill(4)(false.B))
        c.coalReqValid.expect(false.B)
      }
  }

//  it should "not touch uncoalescable requests" in {}
//
//  it should "allow temporal coalescing when depth >=2" in {}
//
//  it should "select the most coverage mono-coalescer" in {}
//
//  it should "resort to the backup policy when coverage is below average" in {}
}

class CoalShiftQueueTest extends AnyFlatSpec with ChiselScalatestTester {
  behavior of "request shift queues"

  def attemptEnqueue(c: CoalShiftQueue[UInt], bits: Seq[UInt], valids: Seq[Bool]): Unit = {
    ((c.io.queue.enq zip bits) zip valids).foreach { case ((enq, ent), valid) =>
      enq.ready.expect(true.B)
      enq.valid.poke(valid)
      enq.bits.poke(ent)
    }
    c.clock.step()
  }

  def expectDequeue(c: CoalShiftQueue[UInt], bits: Seq[UInt], valids: Seq[Bool]): Unit = {
    ((c.io.queue.deq zip bits) zip valids).foreach { case ((deq, ent), valid) =>
      deq.valid.expect(valid)
      deq.bits.expect(ent)
    }
  }

  def pokeVec[T <: Data](vec: Seq[T], value: Seq[T]): Unit = {
    (vec zip value).foreach { case (a, b) => a.poke(b) }
  }

  it should "work like normal shiftqueue when no invalidate" in {

    test(new CoalShiftQueue(UInt(8.W),4, testConfig)) { c =>
      c.io.coalescable.foreach(_.poke(true.B))
      c.io.queue.deq.foreach(_.ready.poke(false.B))

      attemptEnqueue(c, Seq.fill(4)(1.U), Seq.fill(4)(true.B))
      attemptEnqueue(c, Seq.fill(4)(2.U), Seq(true.B, false.B, false.B, false.B)) // should remain synchronous
      attemptEnqueue(c, Seq.fill(4)(3.U), Seq.fill(4)(true.B))

      c.io.queue.enq.foreach(_.valid.poke(false.B))
      c.io.queue.enq.foreach(_.ready.expect(true.B))
      // check if head is the first enqueued item
      expectDequeue(c, Seq.fill(4)(1.U), Seq.fill(4)(false.B))
      c.clock.step()

      c.io.queue.deq.foreach(_.ready.poke(true.B))
      // should not dequeue because all are coalescable
      expectDequeue(c, Seq.fill(4)(1.U), Seq.fill(4)(false.B))
      c.clock.step()

      pokeVec(c.io.coalescable, Seq(false.B, false.B, false.B, true.B))
      // first 3 items should be valid now
      expectDequeue(c, Seq.fill(4)(1.U), Seq(true.B, true.B, true.B, false.B))
      // only dequeue first item - 4th item should not be dequeued since not valid
      pokeVec(c.io.queue.deq.map(_.ready), Seq(true.B, false.B, false.B, true.B))
      c.clock.step()

      // first item should turn invalid
      c.io.coalescable.foreach(_.poke(false.B))
      expectDequeue(c, Seq.fill(4)(1.U), Seq(false.B, true.B, true.B, true.B))
      // now dequeue everything else in the first line
      c.io.queue.deq.foreach(_.ready.poke(true.B))
      c.clock.step()

      // all dequeued && shifted last cycle
      c.io.coalescable.foreach(_.poke(false.B))
      c.io.queue.deq.foreach(_.ready.poke(true.B))
      expectDequeue(c, Seq.fill(4)(2.U), Seq(true.B, false.B, false.B, false.B))
      c.clock.step()

      pokeVec(c.io.coalescable, Seq(true.B, false.B, true.B, true.B))
      expectDequeue(c, Seq.fill(4)(3.U), Seq(false.B, true.B, false.B, false.B))
      c.clock.step()

      c.io.coalescable.foreach(_.poke(false.B))
      expectDequeue(c, Seq.fill(4)(3.U), Seq(true.B, false.B, true.B, true.B))
      c.clock.step()

      // empty
      expectDequeue(c, Seq.fill(4)(0.U), Seq.fill(4)(false.B))

      // now enqueue back to full & test back pressure
      c.io.queue.deq.foreach(_.ready.poke(false.B))
      attemptEnqueue(c, Seq.fill(4)(1.U), Seq.fill(4)(true.B))
      pokeVec(c.io.coalescable, Seq(true.B, true.B, true.B, true.B))
      attemptEnqueue(c, Seq.fill(4)(2.U), Seq.fill(4)(true.B))
      attemptEnqueue(c, Seq.fill(4)(3.U), Seq.fill(4)(true.B))
      attemptEnqueue(c, Seq.fill(4)(4.U), Seq.fill(4)(true.B))

      // check full
      c.io.queue.enq.foreach(_.ready.expect(false.B))
      c.clock.step()

      // now indicate this cycle will dequeue everything
      c.io.queue.deq.foreach(_.ready.poke(true.B))

      // should still be full, but allow enqueue
      c.io.coalescable.foreach(_.poke(true.B))
      c.io.queue.enq.foreach(_.ready.expect(false.B)) // check full
      c.io.coalescable.foreach(_.poke(false.B))
      attemptEnqueue(c, Seq.fill(4)(5.U), Seq.fill(4)(true.B))

      expectDequeue(c, Seq.fill(4)(2.U), Seq.fill(4)(true.B))
      c.clock.step()

      attemptEnqueue(c, Seq.fill(4)(6.U), Seq.fill(4)(true.B))
    }
  }

  it should "work when enqueing and dequeueing simultaneously" in {
    test(new CoalShiftQueue(UInt(8.W), 4, testConfig)) { c =>
      c.io.invalidate.valid.poke(false.B)

      c.io.coalescable.foreach(_.poke(true.B))
      c.io.queue.deq.foreach(_.ready.poke(false.B))

      attemptEnqueue(c, Seq.fill(4)(1.U), Seq.fill(4)(true.B))

      // mark for dequeue
      c.io.coalescable.foreach(_.poke(false.B))
      c.io.queue.deq.foreach(_.ready.poke(true.B))
      expectDequeue(c, Seq.fill(4)(1.U), Seq.fill(4)(true.B))
      attemptEnqueue(c, Seq.fill(4)(2.U), Seq.fill(4)(true.B))

      expectDequeue(c, Seq.fill(4)(2.U), Seq.fill(4)(true.B))
      attemptEnqueue(c, Seq.fill(4)(3.U), Seq.fill(4)(true.B))

      expectDequeue(c, Seq.fill(4)(3.U), Seq.fill(4)(true.B))
      attemptEnqueue(c, Seq.fill(4)(4.U), Seq.fill(4)(true.B))

      expectDequeue(c, Seq.fill(4)(4.U), Seq.fill(4)(true.B))
      attemptEnqueue(c, Seq.fill(4)(5.U), Seq.fill(4)(true.B))

      // disable dequeue
      c.io.queue.deq.foreach(_.ready.poke(false.B))
      expectDequeue(c, Seq.fill(4)(5.U), Seq.fill(4)(true.B))
      attemptEnqueue(c, Seq.fill(4)(6.U), Seq.fill(4)(true.B))
      attemptEnqueue(c, Seq.fill(4)(7.U), Seq.fill(4)(true.B))
      attemptEnqueue(c, Seq.fill(4)(8.U), Seq.fill(4)(true.B))

      c.io.queue.enq.foreach(_.ready.expect(false.B))
      c.io.queue.deq.foreach(_.ready.poke(true.B))
      expectDequeue(c, Seq.fill(4)(5.U), Seq.fill(4)(true.B))
      c.clock.step()
      expectDequeue(c, Seq.fill(4)(6.U), Seq.fill(4)(true.B))
      c.clock.step()
      expectDequeue(c, Seq.fill(4)(7.U), Seq.fill(4)(true.B))
      c.clock.step()
      expectDequeue(c, Seq.fill(4)(8.U), Seq.fill(4)(true.B))
      c.clock.step()
    }
  }
/*
  it should "work when enqueing and dequeueing simultaneously to a depth=1 queue" in {
    test(new CoalShiftQueue(UInt(8.W), 1)) { c =>
      c.io.invalidate.valid.poke(false.B)
      c.io.allowShift.poke(true.B)

      // prepare
      c.io.queue.deq.ready.poke(true.B)
      c.io.queue.enq.ready.expect(true.B)
      c.io.queue.enq.valid.poke(true.B)
      c.io.queue.enq.bits.poke(0x12.U)
      c.clock.step()
      // enqueue and dequeue simultaneously
      c.io.queue.deq.ready.poke(true.B)
      c.io.queue.enq.ready.expect(true.B)
      c.io.queue.enq.valid.poke(true.B)
      c.io.queue.enq.bits.poke(0x34.U)
      c.io.queue.deq.valid.expect(true.B)
      c.io.queue.deq.bits.expect(0x12.U)
      c.clock.step()
      // enqueue and dequeue simultaneously once more
      c.io.queue.deq.ready.poke(true.B)
      c.io.queue.enq.ready.expect(true.B)
      c.io.queue.enq.valid.poke(true.B)
      c.io.queue.enq.bits.poke(0x56.U)
      c.io.queue.deq.valid.expect(true.B)
      c.io.queue.deq.bits.expect(0x34.U)
      c.clock.step()
      // dequeueing back-to-back should work without any holes in the middle
      c.io.queue.deq.ready.poke(true.B)
      c.io.queue.enq.valid.poke(false.B)
      c.io.queue.deq.valid.expect(true.B)
      c.io.queue.deq.bits.expect(0x56.U)
      c.clock.step()
      // make sure is empty
      c.io.queue.deq.ready.poke(true.B)
      c.io.queue.enq.valid.poke(false.B)
      c.io.queue.deq.valid.expect(false.B)
    }
  }

  it should "work when invalidating and enqueueing to a depth=1 queue" in {
    test(new CoalShiftQueue(UInt(8.W), 1)) { c =>
      c.io.invalidate.valid.poke(false.B)
      c.io.allowShift.poke(true.B)
      // no dequeueing
      c.io.queue.deq.ready.poke(false.B)

      // prepare
      c.io.queue.enq.ready.expect(true.B)
      c.io.queue.enq.valid.poke(true.B)
      c.io.queue.enq.bits.poke(0x12.U)
      c.clock.step()
      // invalidate, but don't allow shift
      c.io.allowShift.poke(false.B)
      c.io.invalidate.valid.poke(true.B)
      c.io.invalidate.bits.poke(0x1.U)
      // TODO: we might be able to enqueue to a full depth=1 queue whose only
      // entry just got invalidated, so that enq.ready is true here, but
      // it is a niche case
      c.io.queue.enq.ready.expect(false.B)
      c.clock.step()
      // now try enqueueing now that we have space
      c.io.allowShift.poke(true.B)
      c.io.invalidate.valid.poke(false.B)
      c.io.queue.enq.ready.expect(true.B)
      c.io.queue.enq.valid.poke(true.B)
      c.io.queue.enq.bits.poke(0x34.U)
      c.io.queue.deq.valid.expect(false.B)
      c.clock.step()
      // see if it comes out right next cycle
      c.io.queue.enq.valid.poke(false.B)
      c.io.queue.deq.ready.poke(true.B)
      c.io.queue.deq.valid.expect(true.B)
      c.io.queue.deq.bits.expect(0x34.U)
    }
  }

  it should "invalidate head that is also being dequeued" in {
    test(new CoalShiftQueue(UInt(8.W), 4)) { c =>
      c.io.invalidate.valid.poke(false.B)
      c.io.allowShift.poke(true.B)

      // prepare
      c.io.queue.deq.ready.poke(false.B)
      c.io.queue.enq.ready.expect(true.B)
      c.io.queue.enq.valid.poke(true.B)
      c.io.queue.enq.bits.poke(0x12.U)
      c.clock.step()
      c.io.queue.deq.ready.poke(false.B)
      c.io.queue.enq.ready.expect(true.B)
      c.io.queue.enq.valid.poke(true.B)
      c.io.queue.enq.bits.poke(0x34.U)
      c.clock.step()
      c.io.queue.enq.valid.poke(false.B)

      // invalidate should work for the head just being dequeued at the same
      // cycle
      c.io.invalidate.valid.poke(true.B)
      c.io.invalidate.bits.poke(0x1.U)
      c.io.queue.deq.ready.poke(true.B)
      c.io.queue.deq.valid.expect(false.B)
      c.clock.step()
      // 0x12 should have been dequeued
      c.io.invalidate.valid.poke(false.B)
      c.io.queue.deq.ready.poke(true.B)
      c.io.queue.deq.valid.expect(true.B)
      c.io.queue.deq.bits.expect(0x34.U)
    }
  }

  it should "dequeue invalidated head on its own when allowShift" in {
    test(new CoalShiftQueue(gen = UInt(8.W), entries = 4)) { c =>
      c.io.invalidate.valid.poke(false.B)

      c.io.allowShift.poke(true.B)

      // prepare
      c.io.queue.deq.ready.poke(false.B)
      c.io.queue.enq.ready.expect(true.B)
      c.io.queue.enq.valid.poke(true.B)
      c.io.queue.enq.bits.poke(0x12.U)
      c.clock.step()
      c.io.queue.enq.ready.expect(true.B)
      c.io.queue.enq.valid.poke(true.B)
      c.io.queue.enq.bits.poke(0x34.U)
      c.clock.step()
      c.io.queue.enq.ready.expect(true.B)
      c.io.queue.enq.valid.poke(true.B)
      c.io.queue.enq.bits.poke(0x56.U)
      c.clock.step()
      c.io.queue.enq.valid.poke(false.B)

      // invalidate two entries at head
      c.io.invalidate.valid.poke(true.B)
      c.io.invalidate.bits.poke(0x3.U)
      c.io.queue.deq.ready.poke(false.B)
      // [ 0x56 | 0x34(inv) | 0x12(inv) ]
      c.clock.step()
      //             [ 0x56 | 0x34(inv) ]
      c.io.invalidate.valid.poke(false.B)
      c.io.queue.deq.ready.poke(false.B)
      c.clock.step()
      //                         [ 0x56 ]
      c.io.queue.deq.ready.poke(true.B)
      c.io.queue.deq.valid.expect(true.B)
      c.io.queue.deq.bits.expect(0x56.U)
      c.clock.step()
      c.io.queue.deq.ready.poke(true.B)
      c.io.queue.deq.valid.expect(false.B)
      c.clock.step()

      // do one more enqueue-then-dequeue to see if used bit was properly cleared
      c.io.queue.deq.ready.poke(false.B)
      c.io.queue.enq.ready.expect(true.B)
      c.io.queue.enq.valid.poke(true.B)
      c.io.queue.enq.bits.poke(0x78.U)
      c.clock.step()
      // should dequeue right away
      c.io.queue.enq.valid.poke(false.B)
      c.io.queue.deq.ready.poke(true.B)
      c.io.queue.deq.valid.expect(true.B)
      c.io.queue.deq.bits.expect(0x78.U)
    }
  }

  it should "overwrite invalidated tail when enqueuing" in {
    test(new CoalShiftQueue(UInt(8.W), 4)) { c =>
      c.io.invalidate.valid.poke(false.B)
      c.io.invalidate.bits.poke(0.U)
      c.io.allowShift.poke(true.B)

      // prepare
      c.io.queue.deq.ready.poke(false.B)
      c.io.queue.enq.ready.expect(true.B)
      c.io.queue.enq.valid.poke(true.B)
      c.io.queue.enq.bits.poke(0x12.U)
      c.clock.step()
      // invalidate and enqueue at the tail at the same time
      c.io.invalidate.valid.poke(true.B)
      c.io.invalidate.bits.poke(0x1.U)
      c.io.queue.deq.ready.poke(false.B)
      c.io.queue.enq.ready.expect(true.B)
      c.io.queue.enq.valid.poke(true.B)
      c.io.queue.enq.bits.poke(0x34.U)
      c.clock.step()
      c.io.invalidate.valid.poke(false.B)
      c.io.queue.enq.valid.poke(false.B)
      // now should be able to dequeue immediately as tail is overwritten
      c.io.queue.deq.ready.poke(true.B)
      c.io.queue.deq.valid.expect(true.B)
      c.io.queue.deq.bits.expect(0x34)
    }
  }*/
}

// class UncoalescerUnitTest extends AnyFlatSpec with ChiselScalatestTester {
//   behavior of "uncoalescer"
//   object uncoalescerTestConfig extends CoalescerConfig(
//     enable = true,
//     numLanes = 4,
//     queueDepth = 2,
//     waitTimeout = 8,
//     addressWidth = 24,
//     dataBusWidth = 4, // 128 bit data bus
//     wordSizeInBytes = 4,
//     numOldSrcIds = 16,
//     numNewSrcIds = 4,
//     respQueueDepth = 4,
//     coalLogSizes = Seq(4),
//     sizeEnum = DefaultInFlightTableSizeEnum,
//     numCoalReqs = 1,
//     numArbiterOutputPorts = 4,
//     bankStrideInBytes = 64,
//   )

//   val config = uncoalescerTestConfig

//   val nonCoalReqT = new NonCoalescedRequest(config)
//   val coalReqT = new CoalescedRequest(config)

//   it should "work in general case" in {
//     test(new Uncoalescer(config, nonCoalReqT, coalReqT))
//     // vcs helps with simulation time, but sometimes errors with
//     // "mutation occurred during iteration" java error
//     // .withAnnotations(Seq(VcsBackendAnnotation))
//     { c =>
//       // 4 lanes, queue depth 2
//       c.io.windowElts(0)(0).op.poke(0.U)
//       c.io.windowElts(0)(0).source.poke(1.U)
//       c.io.windowElts(0)(0).address.poke(0x4.U)
//       c.io.windowElts(0)(0).size.poke(2.U)
//       c.io.windowElts(0)(1).op.poke(0.U)
//       c.io.windowElts(0)(1).source.poke(2.U)
//       c.io.windowElts(0)(1).address.poke(0x4.U) // two reqs from one lane
//       c.io.windowElts(0)(1).size.poke(2.U)
//       c.io.windowElts(2)(0).op.poke(0.U)
//       c.io.windowElts(2)(0).source.poke(2.U)
//       c.io.windowElts(2)(0).address.poke(0x8.U)
//       c.io.windowElts(2)(0).size.poke(2.U)
//       c.io.windowElts(2)(1).op.poke(0.U)
//       c.io.windowElts(2)(1).source.poke(2.U)
//       c.io.windowElts(2)(1).address.poke(0xc.U)
//       c.io.windowElts(2)(1).size.poke(2.U)
//       // indicate lane 0 and 2 are used for coalescing
//       c.io.invalidate.valid.poke(true.B)
//       c.io.invalidate.bits(0).poke(0x3.U) // 2'b11 for depth=2
//       c.io.invalidate.bits(1).poke(0x0.U)
//       c.io.invalidate.bits(2).poke(0x3.U)
//       c.io.invalidate.bits(3).poke(0x0.U)

//       val sourceId = 0.U
//       c.io.coalReq.valid.poke(true.B)
//       c.io.coalReq.bits.source.poke(sourceId)
//       c.io.coalReq.ready.expect(true.B)

//       c.clock.step()

//       c.io.coalReq.valid.poke(false.B)
//       c.io.invalidate.valid.poke(false.B)

//       c.clock.step()

//       c.io.coalResp.valid.poke(true.B)
//       c.io.coalResp.bits.source.poke(sourceId)
//       val lit = (BigInt(0x0123456789abcdefL) << 64) | BigInt(0x5ca1ab1edeadbeefL)
//       // val lit = BigInt(0x0123456789abcdefL)
//       c.io.coalResp.bits.data.poke(lit.U)

//       // table lookup is combinational at the same cycle
//       c.io.uncoalResps(0)(0).valid.expect(true.B)
//       c.io.uncoalResps(1)(0).valid.expect(false.B)
//       c.io.uncoalResps(2)(0).valid.expect(true.B)
//       c.io.uncoalResps(3)(0).valid.expect(false.B)

//       // offset is counting from LSB
//       c.io.uncoalResps(0)(0).bits.data.expect(0x5ca1ab1eL.U)
//       c.io.uncoalResps(0)(0).bits.source.expect(1.U)
//       c.io.uncoalResps(0)(1).bits.data.expect(0x5ca1ab1eL.U)
//       c.io.uncoalResps(0)(1).bits.source.expect(2.U)
//       c.io.uncoalResps(2)(0).bits.data.expect(0x89abcdefL.U)
//       c.io.uncoalResps(2)(0).bits.source.expect(2.U)
//       c.io.uncoalResps(2)(1).bits.data.expect(0x01234567L.U)
//       c.io.uncoalResps(2)(1).bits.source.expect(2.U)
//     }
//   }

//   it should "uncoalesce when coalesced to the same word offset" in {
//     test(new Uncoalescer(config, nonCoalReqT, coalReqT))
//     // .withAnnotations(Seq(VcsBackendAnnotation))
//     { c =>
//       // 4 lanes, queue depth 2
//       c.io.windowElts(0)(0).op.poke(0.U)
//       c.io.windowElts(0)(0).source.poke(0.U)
//       c.io.windowElts(0)(0).address.poke(0x4.U)
//       c.io.windowElts(0)(0).size.poke(2.U)
//       c.io.windowElts(1)(0).op.poke(0.U)
//       c.io.windowElts(1)(0).source.poke(1.U)
//       c.io.windowElts(1)(0).address.poke(0x4.U) // two reqs from one lane
//       c.io.windowElts(1)(0).size.poke(2.U)
//       c.io.windowElts(2)(0).op.poke(0.U)
//       c.io.windowElts(2)(0).source.poke(2.U)
//       c.io.windowElts(2)(0).address.poke(0x4.U)
//       c.io.windowElts(2)(0).size.poke(2.U)
//       c.io.windowElts(3)(0).op.poke(0.U)
//       c.io.windowElts(3)(0).source.poke(3.U)
//       c.io.windowElts(3)(0).address.poke(0x4.U)
//       c.io.windowElts(3)(0).size.poke(2.U)
//       // indicate lanes used for coalescing
//       c.io.invalidate.valid.poke(true.B)
//       c.io.invalidate.bits(0).poke(0x1.U) // 2'b01 for enabling head
//       c.io.invalidate.bits(1).poke(0x1.U)
//       c.io.invalidate.bits(2).poke(0x1.U)
//       c.io.invalidate.bits(3).poke(0x1.U)

//       val sourceId = 0.U
//       c.io.coalReq.valid.poke(true.B)
//       c.io.coalReq.bits.source.poke(sourceId)
//       c.io.coalReq.ready.expect(true.B)

//       c.clock.step()

//       c.io.coalReq.valid.poke(false.B)
//       c.io.invalidate.valid.poke(false.B)

//       c.clock.step()

//       c.io.coalResp.valid.poke(true.B)
//       c.io.coalResp.bits.source.poke(sourceId)
//       val lit = (BigInt(0x0123456789abcdefL) << 64) | BigInt(0x5ca1ab1edeadbeefL)
//       c.io.coalResp.bits.data.poke(lit.U)

//       // table lookup is combinational at the same cycle
//       // offset is counting from LSB
//       c.io.uncoalResps(0)(0).valid.expect(true.B)
//       c.io.uncoalResps(0)(0).bits.data.expect(0x5ca1ab1eL.U)
//       c.io.uncoalResps(0)(0).bits.source.expect(0.U)
//       c.io.uncoalResps(0)(1).valid.expect(false.B)
//       c.io.uncoalResps(1)(0).valid.expect(true.B)
//       c.io.uncoalResps(1)(0).bits.data.expect(0x5ca1ab1eL.U)
//       c.io.uncoalResps(1)(0).bits.source.expect(1.U)
//       c.io.uncoalResps(1)(1).valid.expect(false.B)
//       c.io.uncoalResps(2)(0).valid.expect(true.B)
//       c.io.uncoalResps(2)(0).bits.data.expect(0x5ca1ab1eL.U)
//       c.io.uncoalResps(2)(0).bits.source.expect(2.U)
//       c.io.uncoalResps(2)(1).valid.expect(false.B)
//       c.io.uncoalResps(3)(0).valid.expect(true.B)
//       c.io.uncoalResps(3)(0).bits.data.expect(0x5ca1ab1eL.U)
//       c.io.uncoalResps(3)(0).bits.source.expect(3.U)
//       c.io.uncoalResps(3)(1).valid.expect(false.B)
//     }
//   }
// }
