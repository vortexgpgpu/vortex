package radiance.tile

import chisel3._
import chisel3.util._
import org.chipsalliance.diplomacy.lazymodule._
import org.chipsalliance.cde.config.Parameters
import freechips.rocketchip.tilelink._
import freechips.rocketchip.diplomacy.{AddressSet, TransferSizes}
import gemmini.Pipeline
import radiance.subsystem.{RadianceSharedMemKey, TwoPort, TwoReadOneWrite}
import radiance.memory._

import scala.collection.mutable.ArrayBuffer

abstract class RadianceSmemNodeProvider {
  val uniformRNodes: Seq[Seq[Seq[TLNexusNode]]]
  val uniformWNodes: Seq[Seq[Seq[TLNexusNode]]]
  val nonuniformRNodes: Seq[TLNode]
  val nonuniformWNodes: Seq[TLNode]
  val clBusClients: Seq[TLNode]
}

abstract class RadianceSmemNodeProviderImp[T <: RadianceSmemNodeProvider](val outer: T) {}

class RadianceSharedMem[T <: RadianceSmemNodeProvider](
    provider: () => T,
    val providerImp: Option[(T) => RadianceSmemNodeProviderImp[T]],
    clbus: TLBusWrapper
  )(implicit p: Parameters) extends LazyModule {
  val smemKey = p(RadianceSharedMemKey).get
  val wordSize = smemKey.wordSize
  val smemBase = smemKey.address
  val smemBanks = smemKey.numBanks
  val smemWidth = smemKey.numWords * smemKey.wordSize
  val smemDepth = smemKey.size / smemWidth / smemBanks
  val smemSubbanks = smemWidth / wordSize
  val smemSize = smemWidth * smemDepth * smemBanks
  val strideByWord = smemKey.strideByWord

  require(isPow2(smemBanks))

  val smNodes = provider()
  val (uniformRNodes, uniformWNodes, nonuniformRNodes, nonuniformWNodes) =
    (smNodes.uniformRNodes, smNodes.uniformWNodes, smNodes.nonuniformRNodes, smNodes.nonuniformWNodes)

  implicit val disableMonitors = smemKey.disableMonitors // otherwise it generate 1k+ different tl monitors

  // collection of read and write managers for each sram (sub)bank
  val smemBankMgrs : Seq[Seq[TLManagerNode]] = if (strideByWord) {
    require(isPow2(smemSubbanks))
    (0 until smemBanks).flatMap { bid =>
      (0 until smemSubbanks).map { wid =>
        Seq.fill(smemKey.memType match {
          case TwoPort => 1
          case TwoReadOneWrite => 2
        })(
          TLManagerNode(Seq(TLSlavePortParameters.v1(
            managers = Seq(TLSlaveParameters.v2(
              name = Some(f"sp_bank${bid}_word${wid}_read_mgr"),
              address = Seq(AddressSet(
                smemBase + (smemDepth * smemWidth * bid) + wordSize * wid,
                smemDepth * smemWidth - smemWidth + wordSize - 1
              )),
              supports = TLMasterToSlaveTransferSizes(
                get = TransferSizes(wordSize, wordSize)),
              fifoId = Some(0)
            )),
            beatBytes = wordSize
          )))
        ) ++ Seq(
          TLManagerNode(Seq(TLSlavePortParameters.v1(
            managers = Seq(TLSlaveParameters.v2(
              name = Some(f"sp_bank${bid}_word${wid}_write_mgr"),
              address = Seq(AddressSet(
                smemBase + (smemDepth * smemWidth * bid) + wordSize * wid,
                smemDepth * smemWidth - smemWidth + wordSize - 1
              )),
              supports = TLMasterToSlaveTransferSizes(
                putFull = TransferSizes(wordSize, wordSize),
                putPartial = TransferSizes(wordSize, wordSize)),
              fifoId = Some(0)
            )),
            beatBytes = wordSize
          )))
        )
      }
    }
  } else {
    (0 until smemBanks).map { bank =>
      Seq.fill(smemKey.memType match {
        case TwoPort => 1
        case TwoReadOneWrite => 2
      })(
        TLManagerNode(Seq(TLSlavePortParameters.v1(
          managers = Seq(TLSlaveParameters.v2(
            name = Some(f"sp_bank${bank}_read_mgr"),
            address = Seq(AddressSet(smemBase + (smemDepth * smemWidth * bank),
              smemDepth * smemWidth - 1)),
            supports = TLMasterToSlaveTransferSizes(
              get = TransferSizes(1, smemWidth)),
            fifoId = Some(0)
          )),
          beatBytes = smemWidth
        )))
      ) ++ Seq(
        TLManagerNode(Seq(TLSlavePortParameters.v1(
          managers = Seq(TLSlaveParameters.v2(
            name = Some(f"sp_bank${bank}_write_mgr"),
            address = Seq(AddressSet(smemBase + (smemDepth * smemWidth * bank),
              smemDepth * smemWidth - 1)),
            supports = TLMasterToSlaveTransferSizes(
              putFull = TransferSizes(1, smemWidth),
              putPartial = TransferSizes(1, smemWidth)),
            fifoId = Some(0)
          )),
          beatBytes = smemWidth
        )))
      )
    }
  }

  val uniformPolicyNodes: Seq[ArrayBuffer[ArrayBuffer[ExtPolicyMasterNode]]] = // mutable
    Seq.fill(2)(ArrayBuffer.fill(smemBanks)(ArrayBuffer.fill(smemSubbanks)(null)))
  val uniformNodesIn: Seq[ArrayBuffer[ArrayBuffer[Seq[TLIdentityNode]]]] =
    Seq.fill(2)(ArrayBuffer.fill(smemBanks)(ArrayBuffer.fill(smemSubbanks)(Seq())))
  val uniformNodesOut: Seq[ArrayBuffer[ArrayBuffer[TLIdentityNode]]] =
    Seq.fill(2)(ArrayBuffer.fill(smemBanks)(ArrayBuffer.fill(smemSubbanks)(null)))

  if (strideByWord) {
    smemBankMgrs.grouped(smemSubbanks).zipWithIndex.foreach { case (bankMgrs, bid) =>
      bankMgrs.zipWithIndex.foreach { case (ports, wid) =>
        val readPorts = ports.init
        val writePort = ports.last

        guardMonitors { implicit p =>
          val urXbar = XbarWithExtPolicy(Some(s"ur_b${bid}_w${wid}"))
          val uwXbar = XbarWithExtPolicy(Some(s"uw_b${bid}_w${wid}"))

          // connect policy nodes
          val rPolicyNode = ExtPolicyMasterNode(uniformRNodes(bid)(wid).length)
          val wPolicyNode = ExtPolicyMasterNode(uniformWNodes(bid)(wid).length)
          urXbar.policySlaveNode := rPolicyNode
          uwXbar.policySlaveNode := wPolicyNode
          uniformPolicyNodes.head(bid)(wid) = rPolicyNode
          uniformPolicyNodes.last(bid)(wid) = wPolicyNode

          // connect clients
          (Seq(urXbar, uwXbar) lazyZip uniformNodesIn lazyZip Seq(uniformRNodes, uniformWNodes))
            .foreach { case (xbar, idBuf, uNodes) =>

              idBuf(bid)(wid) = uNodes(bid)(wid).map { u =>
                val id = TLIdentityNode()
                xbar.node := id := u
                id
              }
            }

          uniformNodesOut.head(bid)(wid) = connectOne(urXbar.node, TLIdentityNode.apply)
          uniformNodesOut.last(bid)(wid) = connectOne(uwXbar.node, TLIdentityNode.apply)

          // connect memory
          smemKey.memType match {
            case TwoPort => {
              val subbankRXbar = TLXbar(TLArbiter.lowestIndexFirst, Some(s"smem_b${bid}_w${wid}_r_xbar"))
              subbankRXbar := uniformNodesOut.head(bid)(wid)
              nonuniformRNodes.foreach( subbankRXbar :=* _ )
              readPorts.head := subbankRXbar
            }
            case TwoReadOneWrite => {
              val subbankRXbars = DoubleOutXbar(Seq(uniformNodesOut.head(bid)(wid)) ++ nonuniformRNodes)
              (readPorts zip subbankRXbars).foreach { case (rp, sbx) => rp := sbx }
            }
          }

          val subbankWXbar = TLXbar(TLArbiter.lowestIndexFirst, Some(s"smem_b${bid}_w${wid}_w_xbar"))
          writePort := subbankWXbar
          subbankWXbar := uniformNodesOut.last(bid)(wid)
          nonuniformWNodes.foreach( subbankWXbar :=* _ )
        }
      }
    }
  } else { // not stride by word
    require(smemKey.memType == TwoPort, "double read ports not implemented")

    val smemRXbar = TLXbar()
    val smemWXbar = TLXbar()

    guardMonitors { implicit p =>
      (uniformRNodes.flatten.flatten ++ nonuniformRNodes).foreach {
        smemRXbar :=* TLWidthWidget(wordSize) :=* _
      }
      (uniformWNodes.flatten.flatten ++ nonuniformWNodes).foreach {
        smemWXbar :=* TLWidthWidget(wordSize) :=* _
      }
    }

    smemBankMgrs.foreach { mem =>
      require(mem.length == 2)
      mem.head := smemRXbar
      mem.last := smemWXbar
    }
  } // stride by word

  guardMonitors { implicit p => smNodes.clBusClients.foreach(clbus.inwardNode := _) }

  lazy val module = new RadianceSharedMemImp(this)
}

class RadianceSharedMemImp[T <: RadianceSmemNodeProvider](outer: RadianceSharedMem[T]) extends LazyModuleImp(outer) {

  val smNodesImp = outer.providerImp.map(impFn => impFn(outer.smNodes))

  case class ReadPort[U <: Data](ren: Bool, data: U)
  case class WritePort[U <: Data](wen: Bool, data: U, mask: UInt)

  def makeReadBuffer[U <: Data](port: ReadPort[U], rNode: TLBundle, rEdge: TLEdgeIn): Unit = {
    port.ren := rNode.a.fire

    val dataPipeIn = Wire(DecoupledIO(port.data.cloneType))
    dataPipeIn.valid := RegNext(port.ren)
    dataPipeIn.bits := port.data

    val metadataPipeIn = Wire(DecoupledIO(new Bundle {
      val source = rNode.a.bits.source.cloneType
      val size = rNode.a.bits.size.cloneType
    }))
    metadataPipeIn.valid := port.ren
    metadataPipeIn.bits.source := rNode.a.bits.source
    metadataPipeIn.bits.size := rNode.a.bits.size

    val sramReadBackupReg = RegInit(0.U.asTypeOf(Valid(port.data.cloneType)))

    val dataPipeInst = Module(new Pipeline(dataPipeIn.bits.cloneType, 1)())
    dataPipeInst.io.in <> dataPipeIn
    val dataPipe = dataPipeInst.io.out
    val metadataPipe = Pipeline(metadataPipeIn, 2)
    assert((dataPipe.valid || sramReadBackupReg.valid) === metadataPipe.valid)

    // data pipe is filled, but D is not ready and SRAM read came back
    when (dataPipe.valid && !rNode.d.ready && dataPipeIn.valid) {
      assert(!dataPipeIn.ready) // we should fill backup reg only if data pipe is not enqueueing
      assert(!sramReadBackupReg.valid) // backup reg should be empty
      assert(!metadataPipeIn.ready) // metadata should be filled previous cycle
      sramReadBackupReg.valid := true.B
      sramReadBackupReg.bits := port.data
    }.otherwise {
      assert(dataPipeIn.ready || !dataPipeIn.valid) // do not skip any response
    }

    assert(metadataPipeIn.fire || !port.ren) // when requesting sram, metadata needs to be ready
    assert(rNode.d.fire === metadataPipe.fire) // metadata dequeues iff D fires

    // when D becomes ready, and data pipe has emptied, time for backup to empty
    when (rNode.d.ready && sramReadBackupReg.valid && !dataPipe.valid) {
      sramReadBackupReg.valid := false.B
    }
    // must empty backup before filling data pipe
    assert(!(sramReadBackupReg.valid && dataPipe.valid && dataPipeIn.fire))

    rNode.d.bits := rEdge.AccessAck(
      Mux(rNode.d.valid, metadataPipe.bits.source, 0.U),
      Mux(rNode.d.valid, metadataPipe.bits.size, 0.U),
      Mux(!dataPipe.valid, sramReadBackupReg.bits, dataPipe.bits).asUInt)
    rNode.d.valid := dataPipe.valid || sramReadBackupReg.valid
    // r node A is not ready only if D is not ready and both slots filled
    rNode.a.ready := rNode.d.ready && !(dataPipe.valid && sramReadBackupReg.valid)
    dataPipe.ready := rNode.d.ready
    metadataPipe.ready := rNode.d.ready
  }

  def makeWriteBuffer[U <: Data](port: WritePort[U], wNode: TLBundle, wEdge: TLEdgeIn): Unit = {
    port.wen := RegNext(wNode.a.fire)
    port.data := RegNext(wNode.a.bits.data)
    port.mask := RegNext(wNode.a.bits.mask)

    val writeResp = Wire(Flipped(wNode.d.cloneType))
    writeResp.bits := wEdge.AccessAck(wNode.a.bits)
    writeResp.valid := wNode.a.valid
    wNode.a.ready := writeResp.ready
    wNode.d <> Queue(writeResp, 2)
  }

  if (outer.strideByWord) {
    val uniformFires = Seq.fill(2)(VecInit.fill(outer.smemBanks)(VecInit.fill(outer.smemSubbanks)(false.B)))

    // instantiate sram banks and connect
    outer.smemBankMgrs.grouped(outer.smemSubbanks).zipWithIndex.foreach { case (bankMgrs, bid) =>

      bankMgrs.zipWithIndex.foreach { case (ports, wid) =>
        val readPorts = ports.init
        val writePort = ports.last

        assert(!readPorts.flatMap(_.portParams.map(_.anySupportPutFull)).reduce(_ || _))
        assert(!writePort.portParams.map(_.anySupportGet).reduce(_ || _))

        val memDepth = outer.smemDepth
        val memWidth = outer.smemWidth
        val wordWidth = outer.wordSize

        outer.smemKey.memType match {
          case TwoPort =>
            val mem = TwoPortSyncMem(
              n = memDepth,
              t = UInt((wordWidth * 8).W),
            )
            // TODO: bring in cluster id
            // mem.suggestName(s"rad_smem_cl${outer.thisClusterParams.clusterId}_b${bid}_w${wid}")

            val (rNode, rEdge) = readPorts.head.in.head
            val (wNode, wEdge) = writePort.in.head

            // address format is
            // [ smem_base | bank_id | line_id | word_id | byte_offset ]
            // line_id is used to index into the SRAMs
            mem.io.raddr := (rNode.a.bits.address & (memDepth * memWidth - 1).U) >> log2Ceil(memWidth).U
            mem.io.waddr := RegNext((wNode.a.bits.address & (memDepth * memWidth - 1).U) >> log2Ceil(memWidth).U)

            assert((bid.U === ((rNode.a.bits.address & (memDepth * memWidth * outer.smemBanks - 1).U) >>
              log2Ceil(memDepth * memWidth).U).asUInt) || !rNode.a.valid, "bank id mismatch with request")
            assert((wid.U === ((rNode.a.bits.address & (memWidth - 1).U) >>
              log2Ceil(wordWidth).U).asUInt) || !rNode.a.valid, "word id mismatch with request")

            makeReadBuffer(ReadPort(mem.io.ren, mem.io.rdata), rNode, rEdge)
            makeWriteBuffer(WritePort(mem.io.wen, mem.io.wdata, mem.io.mask), wNode, wEdge)

          case TwoReadOneWrite =>
            val mem = TwoReadOneWriteSyncMem(
              n = memDepth,
              t = UInt((wordWidth * 8).W),
            )

            val (rNode0, rEdge0) = readPorts.head.in.head
            val (rNode1, rEdge1) = readPorts.last.in.head
            val (wNode, wEdge) = writePort.in.head

            mem.io.raddr0 := (rNode0.a.bits.address & (memDepth * memWidth - 1).U) >> log2Ceil(memWidth).U
            mem.io.raddr1 := (rNode1.a.bits.address & (memDepth * memWidth - 1).U) >> log2Ceil(memWidth).U
            mem.io.waddr := RegNext((wNode.a.bits.address & (memDepth * memWidth - 1).U) >> log2Ceil(memWidth).U)

            makeReadBuffer(ReadPort(mem.io.ren0, mem.io.rdata0), rNode0, rEdge0)
            makeReadBuffer(ReadPort(mem.io.ren1, mem.io.rdata1), rNode1, rEdge1)
            makeWriteBuffer(WritePort(mem.io.wen, mem.io.wdata, mem.io.mask), wNode, wEdge)
        }
      }
    }


    // set up uniform mux selects
    Seq.tabulate(outer.smemBanks) { bid =>
      // note down fire here so the round-robin knows when an input is selected
      Seq.tabulate(outer.smemSubbanks) { wid =>
        (uniformFires zip outer.uniformNodesOut).foreach { case (uf, n) =>
          uf(bid)(wid) := n(bid)(wid).in.head._1.a.fire
        }
      }
      // have a uniform hint to all subbanks in a bank
      val wordSelects1h = Seq(
        Wire(UInt(outer.uniformNodesIn.head(bid).head.length.W)).suggestName(s"ws_r_b${bid}"),
        Wire(UInt(outer.uniformNodesIn.last(bid).head.length.W)).suggestName(s"ws_w_b${bid}"))
      val Seq(validRSources, validWSources) = outer.uniformNodesIn.zipWithIndex.map { case (banks, rw) =>
        VecInit(banks(bid).map(_.map(_.in.head._1.a.valid)).transpose.map { wordsInIdx =>
          VecInit(wordsInIdx.toSeq).asUInt.orR
        }.toSeq).asUInt.suggestName(s"valid_sources_rw${rw}_b${bid}")
      }
      // use round robin to decide uniform select
      (wordSelects1h zip Seq(validRSources, validWSources)).zipWithIndex.foreach { case ((ws, vs), rw) =>
        ws := TLArbiter.roundRobin(vs.getWidth, vs, uniformFires(rw)(bid).asUInt.orR)
      }
      // mask valid into xbar to prevent triggering assertion
      (wordSelects1h lazyZip outer.uniformPolicyNodes lazyZip outer.uniformNodesIn).foreach { case (ws, pn, ui) =>
        (pn(bid) zip ui(bid)).foreach { case (policies, sources) =>
          val inValid = sources.map(_.in.head._1.a.valid)
          val outValid = sources.map(_.out.head._1.a.valid)

          // we mirror the selection in XbarWithExtPolicy
          val hintHit = (ws & VecInit(inValid).asUInt).orR
          val wsActual = Mux(hintHit, ws, TLArbiter.lowestIndexFirst(
            inValid.length, VecInit(inValid).asUInt, hintHit && policies.out.head._1.actual(0)))
          (inValid lazyZip outValid lazyZip wsActual.asBools).foreach { case (iv, ov, sel) =>
            ov := iv && sel // only present output valid if input is selected
          }
        }
      }
      // set policy to use the uniform select as hint
      (outer.uniformPolicyNodes zip wordSelects1h).zipWithIndex.foreach { case ((nodesBw, ws), rw) =>
        nodesBw(bid).foreach { policy =>
          policy.out.head._1.hint := ws
        }
      }
    }

  } else {
    outer.smemBankMgrs.foreach { case Seq(r, w) =>
      val memDepth = outer.smemDepth
      val memWidth = outer.smemWidth

      val mem = TwoPortSyncMem(
        n = memDepth,
        t = UInt((memWidth * 8).W),
      )

      val (rNode, rEdge) = r.in.head
      val (wNode, wEdge) = w.in.head

      mem.io.raddr := (rNode.a.bits.address ^ outer.smemBase.U) >> log2Ceil(memWidth).U
      mem.io.waddr := RegNext((wNode.a.bits.address ^ outer.smemBase.U) >> log2Ceil(memWidth).U)

      makeReadBuffer(ReadPort(mem.io.ren, mem.io.rdata), rNode, rEdge)
      makeWriteBuffer(WritePort(mem.io.wen, mem.io.wdata, mem.io.mask), wNode, wEdge)
    }
  }

  // read/write access counter for smem banks
  val smemAccessesPerCycle = outer.smemBankMgrs.transpose.map { rw =>
    VecInit(rw.map(_.in.head._1.a.fire.asUInt)).reduceTree(_ +& _)
  }
  val smemReadCounter = RegInit(0.U(32.W))
  val smemWriteCounter = RegInit(0.U(32.W))
  smemReadCounter := smemReadCounter +& smemAccessesPerCycle.init.reduce(_ +& _)
  smemWriteCounter := smemWriteCounter +& smemAccessesPerCycle.last
  dontTouch(smemReadCounter)
  dontTouch(smemWriteCounter)

}
