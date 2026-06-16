package radiance.memory

import chisel3._
import chisel3.util._
import freechips.rocketchip.tilelink._
import freechips.rocketchip.util.BundleField
import freechips.rocketchip.diplomacy.AddressSet
import org.chipsalliance.cde.config.Parameters
import org.chipsalliance.diplomacy.ValName
import org.chipsalliance.diplomacy.lazymodule._

// this node splits the incoming requests into two outgoing edges,
// the first edge contains requests that match the filter AddressSet,
// and the second edge contains requests that don't.
// on the return leg, the two responses are arbitrated in a RR fashion.
class AlignFilterNode(filters: Seq[AddressSet])(implicit p: Parameters) extends LazyModule {

  val node = TLNexusNode(clientFn = seq => {
    require(seq.map(_.masters.size).sum == 1, s"there should only be one client to a filter node, " +
      s"found ${seq.map(_.masters.size).sum}")
    val master = seq.head.masters.head

    val inMapping = TLXbar.mapInputIds(Seq.fill(filters.length + 1)(seq.head))
    val unalignedSrcRange = inMapping.last

    seq.head.v1copy(
      clients = filters.zipWithIndex.map { case (filter, i) =>
        master.v2copy(
          name = s"${name}_filter_aligned",
          sourceId = inMapping(i),
          visibility = Seq(filter),
          emits = seq.map(_.anyEmitClaims).reduce(_ mincover _)
        )
      } ++ Seq(
        master.v2copy(
          name = s"${name}_filter_unaligned",
          sourceId = unalignedSrcRange,
          visibility = Seq(AddressSet.everything),
          emits = seq.map(_.anyEmitClaims).reduce(_ mincover _)
        ),
      )
    )
  }, managerFn = seq => {
    val addresses = seq.flatMap(_.slaves.flatMap(_.address))
    val unifiedAddressRange = addresses.flatMap(_.toRanges).sorted.reduce(_.union(_).get)
    assert(isPow2(unifiedAddressRange.size))
    // println(s"$name address range ${unifiedAddressRange}")
    seq.head.v1copy(
      responseFields = BundleField.union(seq.flatMap(_.responseFields)),
      requestKeys = seq.flatMap(_.requestKeys).distinct,
      minLatency = seq.map(_.minLatency).min,
      endSinkId = TLXbar.mapOutputIds(seq).map(_.end).max,
      managers = Seq(TLSlaveParameters.v2(
        name = Some(s"${name}_manager"),
        address = Seq(AddressSet(unifiedAddressRange.base, unifiedAddressRange.size - 1)),
        supports = seq.map(_.anySupportClaims).reduce(_ mincover _)
      ))
    )
  })

  def castD[T <: TLBundleD](d: TLBundleD, targetDType: T): T = {
    val newD = Wire(targetDType.cloneType)
    d.elements.foreach { case (name, data) =>
      val newDField = newD.elements.filter(_._1 == name).head._2
      newDField := data.asTypeOf(newDField)
    }
    newD
  }

  def castD[T <: DecoupledIO[TLBundleD]](ds: Seq[DecoupledIO[TLBundleD]], targetDType: T): Seq[T] = {
    ds.map { d =>
      val newD = Wire(targetDType.cloneType)
      newD.valid := d.valid
      newD.bits := castD(d.bits, targetDType.bits)
      d.ready := newD.ready
      newD
    }
  }

  lazy val module = new LazyModuleImp(this) {
    val (c, cEdge) = node.in.head
    val a = node.out.init.map(_._1)
    val ua = node.out.last._1

    val inMapping = TLXbar.mapInputIds(Seq.fill(filters.length + 1)(node.in.head._2.client))
    val unalignedSrc = inMapping.last

    val aAligned = filters.map(_.contains(c.a.bits.address))

    (a zip aAligned).zipWithIndex.foreach { case ((a, aligned), idx) =>
      a.a.bits := c.a.bits
      a.a.bits.source := inMapping(idx).start.U + c.a.bits.source
      a.a.valid := c.a.valid && aligned
    }
    ua.a.bits := c.a.bits
    ua.a.bits.source := unalignedSrc.start.U + c.a.bits.source // + (1.U << c.a.bits.source.getWidth)
    ua.a.valid := c.a.valid && !aAligned.reduce(_ || _)
    c.a.ready := MuxCase(ua.a.ready, (a zip aAligned).map { case (a, aligned) => aligned -> a.a.ready })

    TLArbiter.robin(cEdge, c.d, castD(a.map(_.d) ++ Seq(ua.d), c.d): _*)
  }
}

object AlignFilterNode {
  def apply(filters: Seq[AddressSet])(implicit p: Parameters, valName: ValName): TLNexusNode = {
    LazyModule(new AlignFilterNode(filters)).node
  }
}
