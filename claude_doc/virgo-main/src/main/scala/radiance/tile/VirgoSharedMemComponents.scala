package radiance.tile

import chisel3._
import chisel3.util._
import org.chipsalliance.diplomacy.lazymodule._
import org.chipsalliance.diplomacy.{DisableMonitors, ValName}
import org.chipsalliance.cde.config.Parameters
import radiance.memory._
import freechips.rocketchip.tilelink._
import freechips.rocketchip.diplomacy.{AddressSet, BufferParams}
import freechips.rocketchip.subsystem.BaseClusterParams
import radiance.subsystem.{CoreSerialized, FullySerialized, NotSerialized, RadianceSharedMemKey}
import gemmini._
import scala.collection.mutable.ArrayBuffer

// virgo-specific tilelink nodes
// generic smem implementation is in RadianceSharedMem.scala
class VirgoSharedMemComponents(
  clusterParams: BaseClusterParams,
  gemminiTiles: Seq[GemminiTile],
  radianceTiles: Seq[RadianceTile],
)(implicit p: Parameters) extends RadianceSmemNodeProvider  {
  val smemKey = p(RadianceSharedMemKey).get
  val wordSize = smemKey.wordSize
  val smemBase = smemKey.address
  val smemBanks = smemKey.numBanks
  val smemWidth = smemKey.numWords * smemKey.wordSize
  val smemDepth = smemKey.size / smemWidth / smemBanks
  val smemSubbanks = smemWidth / wordSize
  val smemSize = smemWidth * smemDepth * smemBanks

  val numCores = radianceTiles.length
  val numLanes = radianceTiles.head.numLsuLanes

  val gemminis = gemminiTiles.map(_.gemmini)
  val gemminiConfigs = gemminis.map(_.config)
  gemminiConfigs.foreach { config =>
    assert(smemBanks == config.sp_banks && isPow2(smemBanks / config.sp_banks))
    assert(smemWidth >= (config.sp_width / 8) && isPow2(smemWidth / (config.sp_width / 8)))
    assert(smemSize == config.sp_capacity.asInstanceOf[CapacityInKilobytes].kilobytes * 1024)
  }
  if (gemminiConfigs.length > 1) {
    if (!(gemminiConfigs.tail.map(_.inputType == gemminiConfigs.head.inputType).reduce(_ && _))) {
      println("******** WARNING ********\n******** gemmini data types do not match\n******** WARNING ********")
    }
  }

  val strideByWord = smemKey.strideByWord
  val filterAligned = smemKey.filterAligned
  val serializeUnaligned = smemKey.serializeUnaligned
  implicit val disableMonitors: Boolean = smemKey.disableMonitors // otherwise it generate 1k+ different tl monitors

  val radianceSmemFanout = radianceTiles.zipWithIndex.flatMap { case (tile, cid) =>
    tile.smemNodes.zipWithIndex.map { case (m, lid) =>
      val smemFanoutXbar = LazyModule(new TLXbar())
      smemFanoutXbar.suggestName(f"rad_smem_fanout_cl${clusterParams.clusterId}_c${cid}_l${lid}_xbar")
      smemFanoutXbar.node :=* m
      smemFanoutXbar.node
    }
  }
  val tcNodeFanouts = radianceTiles.flatMap(_.tcSmemNodes)
    // .map(connectOne(_, () => TLBuffer(BufferParams(2, false, false), BufferParams(0))))
    .map(connectOne(_, () => TLFIFOFixer()))
    .map(connectXbarName(_, Some("tc_fanout")))
  val clBusClients: Seq[TLNode] = radianceSmemFanout

  // convert to monad (very fancy)
  val coreSerialOpt: Option[Unit] = serializeUnaligned match {
    case CoreSerialized => Some(())
    case _ => None
  }

  // uniform mux select for selecting lanes from a single core in unison
  val coreSerialPolicy = coreSerialOpt.map(_ => Seq.fill(2)(Seq.fill(numLanes)(ExtPolicyMasterNode(numCores))))
  val laneSerialXbars = coreSerialOpt.map(_ => Seq.tabulate(2) { rw =>
    Seq.tabulate(numLanes) { lid =>
      XbarWithExtPolicyNoFallback(Some(f"lane_${lid}_serial_in_xbar_$rw"))
    }
  })

  override val (uniformRNodes, uniformWNodes, nonuniformRNodes, nonuniformWNodes) =

  if (strideByWord) {
    def distAndDuplicate(nodes: Seq[TLNode], suffix: String): Seq[Seq[TLNexusNode]] = {
      val wordFanoutNodes = gemminis.zip(nodes).zipWithIndex.map { case ((gemmini, node), gemminiIdx) =>
        val spWidthBytes = gemmini.config.sp_width / 8
        val spSubbanks = spWidthBytes / wordSize
        val dist = DistributorNode(from = spWidthBytes, to = wordSize)
        guardMonitors { implicit p =>
          dist := node
        }
        val fanout = Seq.tabulate(spSubbanks) { w =>
          val buf = TLBuffer(BufferParams(2, false, false), BufferParams(0))
          buf := dist
          connectXbarName(buf, Some(s"spad_g${gemminiIdx}w${w}_fanout_$suffix"))
        }
        Seq.fill(smemWidth / spWidthBytes)(fanout).flatten // smem wider than spad, duplicate masters
      }
      if (nodes.isEmpty) {
        Seq.fill(smemSubbanks)(Seq())
      } else {
        // (gemmini, word) => (word, gemmini)
        wordFanoutNodes.transpose
      }
    }

    // (banks, subbanks, gemminis)
    val spadReadNodes = Seq.fill(smemBanks)(distAndDuplicate(gemminis.map(_.spad_read_nodes), "r"))
    val spadWriteNodes = Seq.fill(smemBanks)(distAndDuplicate(gemminis.map(_.spad_write_nodes), "w"))
    val spadSpWriteNodesSingleBank = distAndDuplicate(gemminis.map(_.spad.spad_writer.node), "ws")
    val spadSpWriteNodes = Seq.fill(smemBanks)(spadSpWriteNodesSingleBank) // executed only once

    // tensor core read nodes
    val tcDistNodes = Seq.fill(smemBanks)(tcNodeFanouts.map(connectOne(_, () => DistributorNode(smemWidth, wordSize))))
    val tcNodes = tcDistNodes.map { tcBank =>
      Seq.fill(smemSubbanks)(tcBank.map(connectOne(_,
        () => TLBuffer(BufferParams(2, false, false)))).map(connectXbarName(_, Some("tc_dist_fanout"))))
    } // (banks, subbanks, tc client)

    val unalignedRWNodes: ArrayBuffer[ArrayBuffer[TLNexusNode]] = // mutable for readability
      ArrayBuffer.fill(numLanes)(ArrayBuffer.fill(numCores)(null))

    if (filterAligned) {
      val numLaneDupes = Math.max(1, smemSubbanks / numLanes)
      val filterRange = Math.min(smemSubbanks, numLanes)

      // (subbank, sources) = rw node
      val fAligned = if (numLanes >= smemSubbanks) {
        val filterNodes: Seq[Seq[TLNode]] = Seq.tabulate(filterRange) { wid =>
          val address = AddressSet(smemBase + wordSize * wid, (smemSize - 1) - (smemSubbanks - 1) * wordSize)

          radianceSmemFanout.grouped(numLanes).toList.zipWithIndex.flatMap { case (lanes, cid) =>
            lanes.zipWithIndex.flatMap { case (lane, lid) =>
              if ((lid % filterRange) == wid) {
                val filterNode = AlignFilterNode(Seq(address))(p, ValName(s"filter_l${lid}_w${wid}"))
                DisableMonitors { implicit p => filterNode := lane }

                val alignedSplitter = Seq(connectOne(filterNode, () =>
                  RWSplitterNode(address, s"aligned_splitter_c${cid}_l${lid}_w${wid}")))

                unalignedRWNodes(lid)(cid) = connectOne(filterNode, () =>
                  RWSplitterNode(AddressSet.everything, s"unaligned_splitter_c${cid}_l${lid}"))

                alignedSplitter
              } else Seq()
            }
          }
        }

        Seq.fill(2)(filterNodes.map(_.map(connectXbarName(_, Some("rad_aligned")))))
      } else { // aligned: (subbanks, cores) = rw node
        // (lanes, cores) = filter_node
        val filterNodes = Seq.tabulate(filterRange) { wid =>
          val addresses = Seq.tabulate(numLaneDupes) { did =>
            AddressSet(smemBase + (did * filterRange + wid) * wordSize,
              (smemSize - 1) - (smemSubbanks - 1) * wordSize)
          }
          radianceSmemFanout.grouped(numLanes).toSeq.zipWithIndex.map { case (lanes, cid) =>
            val lane = lanes(wid)
            val filterNode = AlignFilterNode(addresses)(p, ValName(s"filter_c${cid}_w${wid}"))
            guardMonitors { implicit p =>
              filterNode := lane
            }
            filterNode
          }
        }
        val fAlignedRW = Seq.tabulate(numLaneDupes) { did =>
          filterNodes.zipWithIndex.map { case (cores, lid) =>
            cores.zipWithIndex.map { case (fn, cid) =>
              val address = AddressSet(smemBase + (did * filterRange + lid) * wordSize,
                (smemSize - 1) - (smemSubbanks - 1) * wordSize)
              connectOne(fn, () => RWSplitterNode(address, s"aligned_split_c${cid}_l${lid}_d${did}"))
            }
          }
        }.flatten
        filterNodes.zipWithIndex.foreach { case (cores, lid) =>
          cores.zipWithIndex.foreach { case (fn, cid) =>
            unalignedRWNodes(lid)(cid) = connectOne(fn, () =>
              RWSplitterNode(AddressSet.everything, s"unaligned_split_c${cid}_l${lid}"))
          }
        }
        Seq.fill(2)(fAlignedRW.map(_.map(connectXbarName(_, Some("rad_aligned")))))
      }

      val fUnaligned: Seq[Seq[TLNode]] = serializeUnaligned match {
        case FullySerialized => Seq.fill(2) {
          val serializedNode = TLEphemeralNode()
          val serializedInXbar = LazyModule(new TLXbar())
          val serializedOutXbar = LazyModule(new TLXbar())
          serializedInXbar.suggestName("unaligned_serialized_in_xbar")
          serializedOutXbar.suggestName("unaligned_serialized_out_xbar")
          guardMonitors { implicit p =>
            unalignedRWNodes.flatten.foreach(serializedInXbar.node := _)
            serializedNode := serializedInXbar.node
            serializedOutXbar.node := serializedNode
          }
          Seq(serializedOutXbar.node)
        }
        case CoreSerialized => Seq.tabulate(2) { rw =>
          // we can either have one core per lane selected (multiple mux selects)
          // or strictly lanes from a single selected core (one mux select). doing the latter here
          unalignedRWNodes.toSeq.zipWithIndex.map { case (coresRW, lid) =>
            val laneSerialXbar = laneSerialXbars.get(rw)(lid)
            laneSerialXbar._1.policySlaveNode := coreSerialPolicy.get(rw)(lid)
            coresRW.foreach(laneSerialXbar._2 := _)
            connectXbarName(connectOne(laneSerialXbar._3, TLEphemeralNode.apply), Some(s"lane_${lid}_serial_out"))
          }
        }
        case NotSerialized => Seq.fill(2)(unalignedRWNodes.toSeq.flatten.map(connectXbar.apply))
      }


      val uniformRNodes: Seq[Seq[Seq[TLNexusNode]]] = (spadReadNodes zip tcNodes).map { case (rb, tcrb) =>
        (rb lazyZip tcrb lazyZip fAligned.head).map { case (rw, tcrw, fa) => rw ++ tcrw ++ fa }
      }
      val uniformWNodes: Seq[Seq[Seq[TLNexusNode]]] = (spadWriteNodes zip spadSpWriteNodes).map { case (wb, wsb) =>
        (wb lazyZip wsb lazyZip fAligned.last).map {
          case (ww, wsw, fa) => ww ++ wsw ++ fa
        }
      }

      // all to all xbar
      val Seq(nonuniformRNodes, nonuniformWNodes) = fUnaligned

      (uniformRNodes, uniformWNodes, nonuniformRNodes, nonuniformWNodes)
    } else {
      val splitterNodes = radianceSmemFanout.map { connectOne(_, () => RWSplitterNode("rad_fanout_splitter")) }
      // these nodes access an entire line simultaneously
      val uniformRNodes: Seq[Seq[Seq[TLNexusNode]]] = spadReadNodes
      val uniformWNodes: Seq[Seq[Seq[TLNexusNode]]] = (spadWriteNodes zip spadSpWriteNodes).map { case (wb, wsb) =>
        (wb zip wsb).map { case (ww, wsw) => ww ++ wsw }
      }
      // random accesses are not serialized here, require so
      require(serializeUnaligned == NotSerialized, "when not filtering, unaligned accesses must be serialized")
      // these nodes are random access
      val nonuniformRNodes: Seq[TLNode] = splitterNodes.map(connectXbarName(_, Some("rad_unaligned_r")))
      val nonuniformWNodes: Seq[TLNode] = splitterNodes.map(connectXbarName(_, Some("rad_unaligned_w")))

      (uniformRNodes, uniformWNodes, nonuniformRNodes, nonuniformWNodes)
    }
  } else { // not stride by word
    val unifiedMemReadNode = TLIdentityNode()
    val unifiedMemWriteNode = TLIdentityNode()

    gemminis.foreach { gemmini =>
      unifiedMemReadNode :=* TLWidthWidget(smemWidth) :=* gemmini.spad_read_nodes
      unifiedMemWriteNode :=* TLWidthWidget(smemWidth) :=* gemmini.spad_write_nodes
      unifiedMemWriteNode := gemmini.spad.spad_writer.node // this is the dma write node
    }

    val splitterNode = RWSplitterNode()
    unifiedMemReadNode := TLWidthWidget(smemWidth) := splitterNode
    unifiedMemWriteNode := TLWidthWidget(smemWidth) := splitterNode

    val coreXbar = TLXbar()
    radianceSmemFanout.foreach(coreXbar := _)
    splitterNode :=* TLWidthWidget(4) :=* coreXbar

    (Seq.empty, Seq.empty, Seq(unifiedMemReadNode), Seq(unifiedMemWriteNode))
  }
}

class VirgoSharedMemComponentsImp[T <: VirgoSharedMemComponents]
  (override val outer: T) extends RadianceSmemNodeProviderImp[T](outer) {

  (outer.laneSerialXbars zip outer.coreSerialPolicy).foreach { case (xbarsRW, policiesRW) =>
    (xbarsRW zip policiesRW).foreach { case (xbars, policies) =>
      // for each lane, if any core is valid
      val coreValids = xbars.map(_._2.in.map(_._1)).transpose.map { core => VecInit(core.map(_.a.valid)).asUInt.orR }
      val select = xbars.map(_._3.in.map(_._1)).transpose.map { core => VecInit(core.map(_.a.fire)).asUInt.orR }
      val coreSelect = TLArbiter.roundRobin(outer.numCores, VecInit(coreValids).asUInt, VecInit(select).asUInt.orR)
      // TODO: roll this into XbarWithExtPolicy
      xbars.foreach { lane =>
        (lane._2.in.map(_._1) lazyZip lane._2.out.map(_._1) lazyZip coreSelect.asBools).foreach { case (li, lo, cs) =>
          lo.a.valid := li.a.valid && cs
        }
      }
      policies.foreach { _.out.head._1.hint := coreSelect }
    }
  }
}
