package radiance.memory
import chisel3._
import chisel3.util._
import midas.targetutils.SynthesizePrintf

// modified from gemmini's two port sync mem
class TwoPortSyncMem[T <: Data](n: Int, t: T, maskedUnitWidth: Int = 8) extends Module {
  val maskWidth = t.getWidth / maskedUnitWidth
  val io = IO(new Bundle {
    val waddr = Input(UInt((log2Ceil(n) max 1).W))
    val raddr = Input(UInt((log2Ceil(n) max 1).W))
    val wdata = Input(t)
    val rdata = Output(t)
    val wen = Input(Bool())
    val ren = Input(Bool())
    val mask = Input(UInt(maskWidth.W))
  })

  when (io.wen && io.ren && io.raddr === io.waddr) {
    SynthesizePrintf(printf("WARNING: read and write collided at address 0x%x\n", io.raddr))
  }

  val maskElem = UInt(maskedUnitWidth.W)
  val memT = Vec(maskWidth, maskElem)
  val mem = SyncReadMem(n, memT, SyncReadMem.WriteFirst)

  io.rdata := mem.read(io.raddr, io.ren).asTypeOf(t)

  when (io.wen) {
    mem.write(io.waddr, io.wdata.asTypeOf(memT), io.mask.asBools)
  }
}

class TwoReadOneWriteSyncMem[T <: Data](n: Int, t: T, maskedUnitWidth: Int = 8) extends Module {
  val maskWidth = t.getWidth / maskedUnitWidth
  val io = IO(new Bundle {
    val waddr = Input(UInt((log2Ceil(n) max 1).W))
    val raddr0 = Input(UInt((log2Ceil(n) max 1).W))
    val raddr1 = Input(UInt((log2Ceil(n) max 1).W))
    val wdata = Input(t)
    val rdata0 = Output(t)
    val rdata1 = Output(t)
    val wen = Input(Bool())
    val ren0 = Input(Bool())
    val ren1 = Input(Bool())
    val mask = Input(UInt(maskWidth.W))
  })

  when (io.wen && io.ren0 && io.raddr0 === io.waddr) {
    SynthesizePrintf(printf("WARNING: read0 and write collided at address 0x%x\n", io.raddr0))
  }
  when (io.wen && io.ren1 && io.raddr1 === io.waddr) {
    SynthesizePrintf(printf("WARNING: read1 and write collided at address 0x%x\n", io.raddr1))
  }

  val maskElem = UInt(maskedUnitWidth.W)
  val memT = Vec(maskWidth, maskElem)
  val mem0 = SyncReadMem(n, memT, SyncReadMem.WriteFirst)
  val mem1 = SyncReadMem(n, memT, SyncReadMem.WriteFirst)

  io.rdata0 := mem0.read(io.raddr0, io.ren0).asTypeOf(t)
  io.rdata1 := mem1.read(io.raddr1, io.ren1).asTypeOf(t)

  when (io.wen) {
    mem0.write(io.waddr, io.wdata.asTypeOf(memT), io.mask.asBools)
    mem1.write(io.waddr, io.wdata.asTypeOf(memT), io.mask.asBools)
  }
}


object TwoPortSyncMem {
  def apply[T <: Data](n: Int, t: T, maskedUnitWidth: Int = 8): TwoPortSyncMem[T] = {
    Module(new TwoPortSyncMem[T](n, t, maskedUnitWidth))
  }
}

object TwoReadOneWriteSyncMem {
  def apply[T <: Data](n: Int, t: T, maskedUnitWidth: Int = 8): TwoReadOneWriteSyncMem[T] = {
    Module(new TwoReadOneWriteSyncMem[T](n, t, maskedUnitWidth))
  }
}
