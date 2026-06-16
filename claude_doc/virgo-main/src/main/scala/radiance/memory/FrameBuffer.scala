package radiance.memory

import chisel3._
import chisel3.util._
import freechips.rocketchip.diplomacy.{AddressSet, SimpleDevice, TransferSizes}
import org.chipsalliance.diplomacy.lazymodule._
import freechips.rocketchip.regmapper.RegField
import freechips.rocketchip.tilelink._
import midas.targetutils.SynthesizePrintf
import org.chipsalliance.cde.config.Parameters

class FrameBuffer(baseAddress: BigInt, width: Int, size: Int, validAddress: BigInt, fbName: String = "fb")
                 (implicit p: Parameters) extends LazyModule {

  val node = TLXbar()

  val bufferNode = TLManagerNode(Seq(TLSlavePortParameters.v1(
    Seq(TLSlaveParameters.v2(
      address = Seq(AddressSet(baseAddress, (1 << log2Ceil(size)) - 1)),
      supports = TLMasterToSlaveTransferSizes(
        putFull = TransferSizes(1, width),
        putPartial = TransferSizes(1, width)
      ),
      fifoId = Some(0))), // requests are handled in order
    beatBytes = width
  )))

  val regDevice = new SimpleDevice("framebuffer-valid-reg", Seq(s"framebuffer-valid-reg"))
  val regNode = TLRegisterNode(
    address = Seq(AddressSet(validAddress, 0x3)), device = regDevice, concurrency = 1)

  bufferNode := TLWidthWidget(4) := TLBuffer() := node
  regNode := TLFragmenter(4, 4) := TLBuffer() := node

  val depth = size >> log2Ceil(width)
  lazy val module = new LazyModuleImp(this) {
    val bufT = Vec(width, UInt(8.W))
    val buf = SyncReadMem(depth, bufT)
    val state = RegInit(false.B) // 0: accepting writes, 1: printing

    val Seq((bufBundle, bufEdge)) = bufferNode.in

    bufBundle.a.ready := !state && bufBundle.d.ready
    bufBundle.d.bits := DontCare
    bufBundle.d.valid := !state && bufBundle.a.valid
    when (bufBundle.a.fire) {
      bufBundle.d.bits := bufEdge.AccessAck(bufBundle.a.bits)
      buf.write(((bufBundle.a.bits.address & (size - 1).U) >> log2Ceil(width)).asUInt,
        bufBundle.a.bits.data.asTypeOf(bufT),
        bufBundle.a.bits.mask.asBools)
    }

    val writeValid = RegInit(0.U(32.W))
    val writeTotal = RegInit(0.U(32.W))
    regNode.regmap(0x00 -> Seq(RegField.w(32, writeValid)))

    // val (writeCounter, writeComplete) = Counter(state.asBool, size / width)
    // when (writeValid(0)) { state := 1.U }
    // when (writeComplete) { state := 0.U }
    val writeCounter = Counter(depth)
    when (writeValid > 0.U) {
      writeValid := 0.U
      writeTotal := writeValid
      state := true.B
      writeCounter.reset()
    }.elsewhen (writeCounter.value === writeTotal - 1.U) {
      state := false.B
    }

    when (state) { writeCounter.inc() }

    val readData = buf.read(writeCounter.value, state)
    val prevIdx = RegNext(writeCounter.value)
    when (RegNext(state)) {
      SynthesizePrintf(printf(s"$fbName %x %x\n", prevIdx, readData.asUInt))
    }
  }
}
