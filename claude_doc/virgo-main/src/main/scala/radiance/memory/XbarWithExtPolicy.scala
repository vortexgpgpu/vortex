package radiance.memory

import chisel3._
import chisel3.experimental.SourceInfo
import freechips.rocketchip.tilelink._
import org.chipsalliance.cde.config.Parameters
import org.chipsalliance.diplomacy.ValName
import org.chipsalliance.diplomacy.lazymodule._
import org.chipsalliance.diplomacy.nodes.{RenderedEdge, SimpleNodeImp, SinkNode, SourceNode}

class ExtPolicyBundle(x: Int) extends Bundle {
  val hint = Output(UInt(x.W))
  val actual = Input(UInt(x.W))
}
object ExtPolicyNodeImp extends SimpleNodeImp[Int, Int, Int, ExtPolicyBundle] {
  def bundle(x: Int) = new ExtPolicyBundle(x)
  def edge(x: Int, y: Int, p: Parameters, sourceInfo: SourceInfo): Int = x
  def render(x: Int): RenderedEdge = RenderedEdge("ffffff")
}
case class ExtPolicyMasterNode(w: Int)(implicit valName: ValName) extends SourceNode(ExtPolicyNodeImp)(Seq(w))
case class ExtPolicySlaveNode()(implicit valName: ValName) extends SinkNode(ExtPolicyNodeImp)(Seq(0))

class XbarWithExtPolicy(nameSuffix: Option[String] = None, useFallback: Boolean = true)
                       (implicit p: Parameters) extends TLXbar(nameSuffix = nameSuffix) {
  val policySlaveNode = ExtPolicySlaveNode()

  class ImplChild extends Impl {
    val policy: TLArbiter.Policy = (width, valids, select) => {
      val in = policySlaveNode.in.head._1
      in.actual := select.asTypeOf(in.actual.cloneType)

      if (useFallback) {
        val hintHit = (valids & in.hint).orR
        val fallback = TLArbiter.lowestIndexFirst(width, valids, !hintHit && select)
        Mux(hintHit, in.hint, fallback)
      } else {
        in.hint
      }
    }
    TLXbar.circuit(policy, node.in, node.out)
  }

  override lazy val module = new ImplChild
}

object XbarWithExtPolicy {
  def apply(nameSuffix: Option[String] = None)
           (implicit p: Parameters): XbarWithExtPolicy = {
    val xbar = LazyModule(new XbarWithExtPolicy(nameSuffix))
    xbar
  }
}

object XbarWithExtPolicyNoFallback {
  def apply(nameSuffix: Option[String] = None)
           (implicit p: Parameters): (XbarWithExtPolicy, TLIdentityNode, TLIdentityNode) = {
    val inIdNode = TLIdentityNode()
    val outIdNode = TLIdentityNode()
    val xbar = LazyModule(new XbarWithExtPolicy(nameSuffix, false))
    xbar.node :=* inIdNode
    outIdNode :=* xbar.node
    (xbar, inIdNode, outIdNode)
  }
}
