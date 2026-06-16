package radiance.memory

import freechips.rocketchip.diplomacy.LazyModule
import freechips.rocketchip.subsystem._
import org.chipsalliance.cde.config.Parameters
import freechips.rocketchip.tilelink._
import radiance.core.{SIMTCoreKey, MemtraceCoreKey}

// TODO: possibly move to somewhere closer to CoalescingUnit
// TODO: separate coalescer config from CanHaveMemtraceCore

// The trait is attached to DigitalTop of Chipyard system, informing it indeed
// has the ability to attach GPU tracer node onto the system bus
trait CanHaveMemtraceCore { this: BaseSubsystem =>
  implicit val p: Parameters

  p(MemtraceCoreKey).map { param =>
    // Safe to use get as WithMemtraceCore requires WithNLanes to be defined
    val simtParam = p(SIMTCoreKey).get
    val config = DefaultCoalescerConfig.copy(
      numLanes = simtParam.nMemLanes, 
      numOldSrcIds = simtParam.nSrcIds
      )
    val numLanes = simtParam.nMemLanes
    val filename = param.tracefilename

    val sbus = locateTLBusWrapper(SBUS)
    // Need to explicitly generate clock domain; see rocket-chip 8881ccd
    val memtracerDomain = sbus.generateSynchronousDomain
    memtracerDomain {
      val tracer = LazyModule(
        new MemTraceDriver(config, filename, param.traceHasSource)(p)
      )
      val coreSideLogger = LazyModule(
        new MemTraceLogger(numLanes, filename, loggerName = "coreside")
      )
      val memSideLogger = LazyModule(
        new MemTraceLogger(numLanes + 1, filename, loggerName = "memside")
      )
      // Must use :=* to ensure the N edges from Tracer doesn't get merged into 1
      // when connecting to SBus
      println(
        s"============ MemTraceDriver instantiated [filename=${param.tracefilename}]"
      )
       val coalescerNode = p(CoalescerKey) match {
         case Some(coalParam) => {
           val coal = LazyModule(new CoalescingUnit(coalParam))
           coal.cpuNode :=* coreSideLogger.node :=* tracer.node // N lanes
           memSideLogger.node :=* coal.aggregateNode            // N+1 lanes
           memSideLogger.node
         }
         case None => tracer.node
       }
      val coalXbar = p(CoalXbarKey) match {
        case Some(xbarParam) =>{
          val coXbar = LazyModule(new TLXbar)
          println(s"============ Using TLXBar for Coalescer Requests ")
          coXbar.node :=* coalescerNode
          coXbar.node
        }
        case None => coalescerNode
      }

      val vortexBank = coalXbar

      //If there is only 1 bank, the code below is useless
      val upstream = p(CoalXbarKey) match {
        case Some(xbarParam) =>{
          val tileXbar = LazyModule(new TLXbar)
          println(s"============ Using TLXBar for L1 Requests ")
          tileXbar.node :=* vortexBank
          tileXbar.node
        }
        case None => vortexBank
      }

      sbus.coupleFrom(s"gpu-tracer") { _ :=* upstream }
    }
  }
}
