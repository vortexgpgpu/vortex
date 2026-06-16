package radiance.memory

import chisel3._
import chisel3.experimental.SourceInfo
import chisel3.util._
import freechips.rocketchip.diplomacy.{AddressSet, TransferSizes, IdRange}
import freechips.rocketchip.tilelink._
import freechips.rocketchip.util.BundleField
import org.chipsalliance.cde.config.Parameters
import org.chipsalliance.diplomacy.ValName
import org.chipsalliance.diplomacy.lazymodule._

class RWSplitterNode(visibility: Option[AddressSet], override val name: String = "rw_splitter")
                    (implicit p: Parameters) extends LazyModule {
  // this node accepts both read and write requests,
  // splits & arbitrates them into one client node per type of operation;
  // there will be N incoming edges, two outgoing edges, with two N:1 muxes;
  // it keeps the read and write channels fully separate to allow parallel processing.
  suggestName(name)
  val node = TLNexusNode(
    clientFn = { seq =>
      val in_mapping = TLXbar.mapInputIds(seq)
      val read_src_range = IdRange(in_mapping.map(_.start).min, in_mapping.map(_.end).max)
      assert((read_src_range.start == 0) && isPow2(read_src_range.end))
      val write_src_range = read_src_range.shift(read_src_range.size)
      val visibilities = seq.flatMap(_.masters.flatMap(_.visibility))
      val unified_vis = if (visibilities.map(_ == AddressSet.everything).reduce(_ || _)) Seq(AddressSet.everything)
        else AddressSet.unify(visibilities)
      // println(s"$name has input visibilities $visibilities, unified to $unified_vis")

      seq.head.v1copy(
        echoFields = BundleField.union(seq.flatMap(_.echoFields)),
        requestFields = BundleField.union(seq.flatMap(_.requestFields)),
        responseKeys = seq.flatMap(_.responseKeys).distinct,
        minLatency = seq.map(_.minLatency).min,
        clients = Seq(
          TLMasterParameters.v1(
            name = s"${name}_read_client",
            sourceId = read_src_range,
            visibility = visibility.map(Seq(_)).getOrElse(unified_vis),
            supportsProbe = TransferSizes.mincover(seq.map(_.anyEmitClaims.get)),
            supportsGet = TransferSizes.mincover(seq.map(_.anyEmitClaims.get)),
            supportsPutFull = TransferSizes.none,
            supportsPutPartial = TransferSizes.none
          ),
          TLMasterParameters.v1(
            name = s"${name}_write_client",
            sourceId = write_src_range,
            visibility = visibility.map(Seq(_)).getOrElse(unified_vis),
            supportsProbe = TransferSizes.mincover(
              seq.map(_.anyEmitClaims.putFull) ++seq.map(_.anyEmitClaims.putPartial)),
            supportsGet = TransferSizes.none,
            supportsPutFull = TransferSizes.mincover(seq.map(_.anyEmitClaims.putFull)),
            supportsPutPartial = TransferSizes.mincover(seq.map(_.anyEmitClaims.putPartial))
          )
        )
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
    val u_out = node.out
    val u_in = node.in
    assert(u_out.length == 2, s"$name should have 2 outgoing edges but has ${u_out.length}")

    val r_out = u_out.head
    val w_out = u_out.last

    val in_src = TLXbar.mapInputIds(u_in.map(_._2.client))
    val in_src_size = in_src.map(_.end).max
    assert(isPow2(in_src_size)) // should be checked already, but just to be sure

    // arbitrate all reads into one read while assigning source prefix, same for write
    val a_arbiter_in = (u_in zip in_src).map { case ((in_node, _), src_range) =>
      val in_r: DecoupledIO[TLBundleA] =
        WireDefault(0.U.asTypeOf(Decoupled(new TLBundleA(in_node.a.bits.params.copy(
          sourceBits = log2Up(in_src_size) + 1
        )))))
      val in_w: DecoupledIO[TLBundleA] = WireDefault(0.U.asTypeOf(in_r.cloneType))

      val req_is_read = in_node.a.bits.opcode === TLMessages.Get

      (Seq(in_r.bits.user, in_r.bits.address, in_r.bits.opcode, in_r.bits.size,
        in_r.bits.mask, in_r.bits.param, in_r.bits.data)
        zip Seq(in_node.a.bits.user, in_node.a.bits.address, in_node.a.bits.opcode, in_node.a.bits.size,
        in_node.a.bits.mask, in_node.a.bits.param, in_node.a.bits.data))
        .foreach { case (x, y) => x := y }
      in_r.bits.source := in_node.a.bits.source | src_range.start.U | Mux(req_is_read, 0.U, in_src_size.U)
      in_w.bits := in_r.bits

      in_r.valid := in_node.a.valid && req_is_read
      in_w.valid := in_node.a.valid && !req_is_read
      in_node.a.ready := Mux(req_is_read, in_r.ready, in_w.ready)

      (in_r, in_w)
    }
    // we cannot use round robin because it might reorder requests, even from the same client
    val (a_arbiter_in_r_nodes, a_arbiter_in_w_nodes) = a_arbiter_in.unzip
    TLArbiter.lowest(r_out._2, r_out._1.a, a_arbiter_in_r_nodes:_*)
    TLArbiter.lowest(w_out._2, w_out._1.a, a_arbiter_in_w_nodes:_*)

    def trim(id: UInt, size: Int): UInt = if (size <= 1) 0.U else id(log2Ceil(size)-1, 0) // from Xbar
    // for each unified mem node client, arbitrate read/write responses on d channel
    (u_in zip in_src).zipWithIndex.foreach { case (((in_node, in_edge), src_range), i) =>
      // assign d channel back based on source, invalid if source prefix mismatch
      val resp = Seq(r_out._1.d, w_out._1.d)
      val source_match = resp.zipWithIndex.map { case (r, i) =>
        (r.bits.source(r.bits.source.getWidth - 1) === i.U(1.W)) && // MSB indicates read(0)/write(1)
          src_range.contains(trim(r.bits.source, in_src_size))
      }
      val d_arbiter_in = resp.map(r => WireDefault(
        0.U.asTypeOf(Decoupled(new TLBundleD(r.bits.params.copy(
          sourceBits = in_node.d.bits.source.getWidth,
          sizeBits = in_node.d.bits.size.getWidth
        ))))
      ))

      (d_arbiter_in lazyZip resp lazyZip source_match).foreach { case (arb_in: DecoupledIO[TLBundleD], r, sm) =>
        (Seq(arb_in.bits.user, arb_in.bits.opcode, arb_in.bits.data, arb_in.bits.param,
          arb_in.bits.sink, arb_in.bits.denied, arb_in.bits.corrupt)
          zip Seq(r.bits.user, r.bits.opcode, r.bits.data, r.bits.param,
          r.bits.sink, r.bits.denied, r.bits.corrupt))
          .foreach { case (x, y) => x := y }
        arb_in.bits.source := trim(r.bits.source, 1 << in_node.d.bits.source.getWidth) // we can trim b/c isPow2(prefix)
        arb_in.bits.size := trim(r.bits.size, 1 << in_node.d.bits.size.getWidth) // FIXME: check truncation

        arb_in.valid := r.valid && sm
        r.ready := arb_in.ready
      }

      TLArbiter.robin(in_edge, in_node.d, d_arbiter_in:_*)
    }
  }
}

object RWSplitterNode {
  def apply()(implicit p: Parameters, valName: ValName, sourceInfo: SourceInfo): TLNexusNode = {
    LazyModule(new RWSplitterNode(None, name = valName.value)).node
  }

  def apply(name: String)
           (implicit p: Parameters, valName: ValName, sourceInfo: SourceInfo): TLNexusNode = {
    LazyModule(new RWSplitterNode(None, name = name)).node
  }

  def apply(visibility: AddressSet)
           (implicit p: Parameters, valName: ValName, sourceInfo: SourceInfo): TLNexusNode = {
    apply(visibility, valName.value)
  }

  def apply(visibility: AddressSet, name: String)
           (implicit p: Parameters, valName: ValName, sourceInfo: SourceInfo): TLNexusNode = {
    LazyModule(new RWSplitterNode(Some(visibility), name = name)).node
  }
}
