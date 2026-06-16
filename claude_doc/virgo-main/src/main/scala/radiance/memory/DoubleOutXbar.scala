package radiance.memory

import chisel3._
import chisel3.util._
import freechips.rocketchip.diplomacy.{AddressSet, TransferSizes, IdRange, BufferParams}
import freechips.rocketchip.tilelink._
import freechips.rocketchip.util.BundleField
import org.chipsalliance.cde.config.Parameters
import org.chipsalliance.diplomacy.ValName
import org.chipsalliance.diplomacy.lazymodule._

class DuplicatorNode(override val name: String = "dup")
                    (implicit p: Parameters) extends LazyModule {
  // tilelink node that has two identical managers for parallelizing request processing
  // one of the two managers must deassert ready when A channel is valid

  val node = TLNexusNode(
    clientFn = { seq =>
      val inMapping = TLXbar.mapInputIds(seq)
      val sourceRange = IdRange(inMapping.map(_.start).min, inMapping.map(_.end).max)
      assert((sourceRange.start == 0) && isPow2(sourceRange.end))

      val visibilities = seq.flatMap(_.masters.flatMap(_.visibility))
      val unifiedVis = if (visibilities.map(_ == AddressSet.everything).reduce(_ || _)) Seq(AddressSet.everything)
      else AddressSet.unify(visibilities)

      seq.head.v1copy(
        echoFields = BundleField.union(seq.flatMap(_.echoFields)),
        requestFields = BundleField.union(seq.flatMap(_.requestFields)),
        responseKeys = seq.flatMap(_.responseKeys).distinct,
        minLatency = seq.map(_.minLatency).min,
        clients = Seq.tabulate(2) { i =>
          TLMasterParameters.v1(
            name = s"${name}_read_client",
            sourceId = sourceRange.shift(sourceRange.size * i),
            visibility = unifiedVis,
            supportsProbe = TransferSizes.mincover(seq.map(_.anyEmitClaims.get)),
            supportsGet = TransferSizes.mincover(seq.map(_.anyEmitClaims.get)),
            supportsPutFull = TransferSizes.mincover(seq.map(_.anyEmitClaims.putFull)),
            supportsPutPartial = TransferSizes.mincover(seq.map(_.anyEmitClaims.putPartial))
          )
        }
      )
    },
    managerFn = { seq =>
      println(f"combined address range of $name managers: " +
        f"${AddressSet.unify(seq.flatMap(_.slaves.flatMap(_.address)))}, supports:" +
        f"${seq.map(_.anySupportClaims).reduce(_ mincover _)}")

      seq.head.v1copy(
        responseFields = BundleField.union(seq.flatMap(_.responseFields)),
        requestKeys = seq.flatMap(_.requestKeys).distinct,
        minLatency = seq.map(_.minLatency).min,
        endSinkId = TLXbar.mapOutputIds(seq).map(_.end).max,
        managers = Seq(TLSlaveParameters.v2(
          name = Some(s"${name}_manager"),
          address = AddressSet.unify(seq.flatMap(_.slaves.flatMap(_.address))),
          supports = seq.map(_.anySupportClaims).reduce(_ mincover _),
          fifoId = Some(0),
        ))
      )
    }
  )

  lazy val module = new LazyModuleImp(this) {
    assert(node.out.length == 2, s"$name should have 2 outgoing edges but has ${node.out.length}")
    assert(node.in.length == 1, s"$name should have one incoming edge but has ${node.in.length}")

    val inSourceWidth = log2Ceil(node.in.head._2.master.endSourceId)
    val inSourceEnd = 1 << inSourceWidth

    val nodeIn = node.in.head._1
    val nodeOuts = node.out.map(_._1)

    val sourceEnq = Wire(DecoupledIO(UInt(inSourceWidth.W)))
    sourceEnq.valid := nodeIn.a.valid && nodeOuts.map(_.a.ready).reduce(_ || _)
    sourceEnq.bits := nodeIn.a.bits.source

    val idQueue = Queue(sourceEnq, entries = 4, pipe = false, flow = false)

    val srcMatch = nodeOuts.map(_.d.bits.source(inSourceWidth - 1, 0) === idQueue.bits)
    idQueue.ready := nodeIn.d.fire

    assert(sourceEnq.fire === nodeIn.a.fire)
    assert(idQueue.fire === nodeIn.d.fire)

    (nodeOuts lazyZip srcMatch lazyZip Seq(0, inSourceEnd)).foreach { case (o, m, p) =>
      o.a.bits := nodeIn.a.bits
      o.a.bits.source := nodeIn.a.bits.source | p.U
      o.a.valid := nodeIn.a.valid
      o.d.ready := nodeIn.d.ready && m
    }
    nodeIn.d.bits := MuxCase(DontCare, (nodeOuts zip srcMatch).map { case (o, m) =>
      m -> o.d.bits
    })
    nodeIn.d.bits.source := MuxCase(DontCare, (nodeOuts zip srcMatch).map { case (o, m) =>
      m -> o.d.bits.source(inSourceWidth - 1, 0)
    })
    nodeIn.d.valid := (nodeOuts zip srcMatch).map { case (o, m) => o.d.valid && m }.reduce(_ || _)
    nodeIn.a.ready := nodeOuts.map(_.a.ready).reduce(_ || _) && sourceEnq.ready

    assert(!(nodeOuts.head.a.ready && nodeOuts.last.a.ready) || !nodeIn.a.valid, "double output fire")
  }
}

object DuplicatorNode {
  def apply()(implicit p: Parameters): TLNexusNode = {
    LazyModule(new DuplicatorNode()).node
  }
}

class DoubleOutXbar(clients: Seq[TLNode], override val name: String = "2o_xbar")
                   (implicit p: Parameters) extends LazyModule {
  val xbar0 = TLXbar(TLArbiter.lowestIndexFirst, Some("double_out_xbar0"))
  val xbar1 = TLXbar(TLArbiter.lowestIndexFirst, Some("double_out_xbar1"))

  implicit val disableMonitors: Boolean = false

  val bufGen = () => TLBuffer(ace = BufferParams(0), bd = BufferParams(2, flow = false, pipe = false))
  val dupedIds = clients.map(connectOne(_, DuplicatorNode.apply)).map { c =>
    val id0 = connectOne(c, TLIdentityNode.apply)
    val id1 = connectOne(c, TLIdentityNode.apply)
    xbar0 := connectOne(id0, bufGen)
    xbar1 := connectOne(id1, bufGen)
    Seq(id0, id1)
  }.transpose

  lazy val module = new LazyModuleImp(this) {
    val id0InReadys = VecInit(dupedIds.head.map(_.in.head._1.a.ready)).asUInt
    val id1InValids = VecInit(dupedIds.last.map(_.in.head._1.a.valid)).asUInt
    val id1OutValids = dupedIds.last.map(_.out.head._1.a.valid)
    val id1InReadys = dupedIds.last.map(_.in.head._1.a.ready)
    val id1OutReadys = VecInit(dupedIds.last.map(_.out.head._1.a.ready)).asUInt
    (id1OutValids zip (id1InValids & (~id0InReadys).asUInt).asBools)
      .foreach { case (o, i) => o := i }
    (id1InReadys zip (id1OutReadys & (~id0InReadys).asUInt).asBools)
      .foreach { case (i, o) => i := o }
  }
}

object DoubleOutXbar {
  def apply(clients: Seq[TLNode])(implicit p: Parameters): Seq[TLNode] = {
    val doubleOutXbar: DoubleOutXbar = LazyModule(new DoubleOutXbar(clients))
    Seq(doubleOutXbar.xbar0, doubleOutXbar.xbar1)
  }
}