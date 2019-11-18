/*******************************************************************************
 HARPtools by Chad D. Kersey, Summer 2011
*******************************************************************************/

#include <iostream>
#include  <iomanip>

// #define USE_DEBUG 7
// #define PRINT_ACTIVE_THREADS

#include "include/types.h"
#include "include/util.h"
#include "include/archdef.h"
#include "include/mem.h"
#include "include/enc.h"
#include "include/core.h"
#include "include/debug.h"

#ifdef EMU_INSTRUMENTATION
#include "include/qsim-harp.h"
#endif


#define NO_MEM_READ  7
#define LB_MEM_READ  0
#define LH_MEM_READ  1
#define LW_MEM_READ  2
#define LBU_MEM_READ 4
#define LHU_MEM_READ 5


#define NO_MEM_WRITE 7
#define SB_MEM_WRITE 0
#define SH_MEM_WRITE 1
#define SW_MEM_WRITE 2

#define INIT_TRACE(trace_inst) \
      trace_inst.valid_inst         = false; \
      trace_inst.pc                 = 0; \
      trace_inst.wid                = schedule_w; \
      trace_inst.rs1                = -1; \
      trace_inst.rs2                = -1; \
      trace_inst.rd                 = -1; \
      trace_inst.is_lw              = false; \
      trace_inst.is_sw              = false; \
      trace_inst.mem_addresses      = new unsigned[a.getNThds()]; \
      for (int tid = 0; tid < a.getNThds(); tid++) trace_inst.mem_addresses[tid] = 0xdeadbeef; \
      trace_inst.mem_stall_cycles   = 0; \
      trace_inst.fetch_stall_cycles = 0; \
      trace_inst.stall_warp         = false; \
      trace_inst.wspawn             = false; \
      trace_inst.stalled            = false;

#define CPY_TRACE(drain, source) \
      drain.valid_inst         = source.valid_inst; \
      drain.pc                 = source.pc; \
      drain.wid                = source.wid; \
      drain.rs1                = source.rs1; \
      drain.rs2                = source.rs2; \
      drain.rd                 = source.rd; \
      drain.is_lw              = source.is_lw; \
      drain.is_sw              = source.is_sw; \
      for (int tid = 0; tid < a.getNThds(); tid++) drain.mem_addresses[tid] = source.mem_addresses[tid]; \
      drain.mem_stall_cycles   = source.mem_stall_cycles; \
      drain.fetch_stall_cycles = source.fetch_stall_cycles; \
      drain.stall_warp         = source.stall_warp; \
      drain.wspawn             = source.wspawn; \
      drain.stalled            = false;

using namespace Harp;
using namespace std;


void printTrace(trace_inst_t * trace, const char * stage_name)
{
    cout << "********************************** " << stage_name << " *********************************\n";
    cout << "valid: " << trace->valid_inst << '\n';
    cout << "PC: " << hex << trace->pc << dec << '\n';
    cout << "wid: " << trace->wid << '\n';
    cout << "rd: " << trace->rd << "\trs1: " << trace->rs1 << "\trs2: " << trace->rs2 << '\n';
    cout << "is_lw: " << trace->is_lw << '\n';
    cout << "is_sw: " << trace->is_sw << '\n';
    cout << "fetch_stall_cycles: " << trace->fetch_stall_cycles << '\n';
    cout << "mem_stall_cycles: " << trace->mem_stall_cycles << '\n';

    cout << "stall_warp: " << trace->stall_warp << '\n';
    cout << "wspawn: " << trace->wspawn << '\n';
    cout << "stalled: " << trace->stalled << '\n';
}

#ifdef EMU_INSTRUMENTATION
void Harp::reg_doRead(Word cpuId, Word regNum) {
  Harp::OSDomain::osDomain->do_reg(cpuId, regNum, 8, true);
}

void Harp::reg_doWrite(Word cpuId, Word regNum) {
  Harp::OSDomain::osDomain->do_reg(cpuId, regNum, 8, false);
}
#endif

Core::Core(const ArchDef &a, Decoder &d, MemoryUnit &mem, Word id):
  a(a), iDec(d), mem(mem), steps(4)
{
  release_warp = false;
  foundSchedule = true;
  schedule_w = 0;
  INIT_TRACE(inst_in_fetch);
  INIT_TRACE(inst_in_decode);
  INIT_TRACE(inst_in_scheduler);
  INIT_TRACE(inst_in_exe);
  INIT_TRACE(inst_in_lsu);
  INIT_TRACE(inst_in_wb);

  for (int i = 0; i < 32; i++)
  {
    stallWarp[i] = false;
    for (int j = 0; j < 32; j++)
    {
        renameTable[i][j] = true;
    }
  }

  cache_simulator = new Vcache_simX;

  m_trace = new VerilatedVcdC;
  cache_simulator->trace(m_trace, 99);
  m_trace->open("simXtrace.vcd");

  cache_simulator->reset = 1;
  cache_simulator->clk   = 0;
  cache_simulator->eval();
  // m_trace->dump(10);
  cache_simulator->reset = 1;
  cache_simulator->clk   = 1;
  cache_simulator->eval();
  // m_trace->dump(11);
  cache_simulator->reset = 0;
  cache_simulator->clk   = 0;

  for (unsigned i = 0; i < a.getNWarps(); ++i)
    w.push_back(Warp(this, i));

  w[0].activeThreads = 1;
  w[0].spawned = true;
}

bool Core::interrupt(Word r0) {
  w[0].interrupt(r0);
}

void Core::step()
{
    cout << "\n\n\n------------------------------------------------------\n";

    steps++;
    cout << "CYCLE: " << steps << '\n';

    cout << "Stalled Warps:\n";
    for (int widd = 0; widd < a.getNWarps(); widd++)
    {
        cout << stallWarp[widd] << " ";
    }
    cout << '\n';

    // cout << "Rename table\n";
    // for (int regii = 0; regii < 32; regii++)
    // {
    //     cout << regii << ": " << renameTable[0][regii] << '\n';
    // }

    cout << '\n';

    this->writeback();
    this->load_store();
    this->execute_unit();
    this->scheduler();
    this->decode();
    this->fetch();

    if (release_warp)
    {
        release_warp = false;
        stallWarp[release_warp_num] = false;
    }
}

void Core::getCacheDelays(trace_inst_t * trace_inst)
{
    static int curr_cycle = 0;
    if (trace_inst->valid_inst)
    {

        bool     in_dcache_in_valid[a.getNThds()];
        unsigned in_dcache_in_address[a.getNThds()];

        unsigned in_dcache_mem_read;
        unsigned in_dcache_mem_write;
        if (trace_inst->is_lw)
        {
            in_dcache_mem_read  = LW_MEM_READ;
            in_dcache_mem_write = NO_MEM_WRITE;
        }
        else if (trace_inst->is_sw)
        {
            in_dcache_mem_read  = NO_MEM_READ;
            in_dcache_mem_write = SW_MEM_WRITE;
        }
        else
        {
            in_dcache_mem_read  = NO_MEM_READ;
            in_dcache_mem_write = NO_MEM_WRITE;
        }

        for (int j = 0; j < a.getNThds(); j++)
        {
            if ((w[trace_inst->wid].tmask[j]) && (trace_inst->is_sw || trace_inst->is_lw))
            {
                in_dcache_in_valid[j]   = true;
                in_dcache_in_address[j] = trace_inst->mem_addresses[j];
            }
            else
            {
                in_dcache_in_valid[j]   = false;
                in_dcache_in_address[j] = 0xdeadbeef;
            }
        }

        cache_simulator->clk = 1;
        cache_simulator->eval();
        m_trace->dump(2*curr_cycle);

        cache_simulator->in_icache_pc_addr       = trace_inst->pc;
        cache_simulator->in_icache_valid_pc_addr = 1;

        // DCache start
        cache_simulator->in_dcache_mem_read  = in_dcache_mem_read;
        cache_simulator->in_dcache_mem_write = in_dcache_mem_write;
        for (int cur_t = 0; cur_t < a.getNThds(); cur_t++)
        {
            cache_simulator->in_dcache_in_valid[cur_t]   = in_dcache_in_valid[cur_t];
            cache_simulator->in_dcache_in_address[cur_t] = in_dcache_in_address[cur_t];
        }
        // DCache end
        cache_simulator->clk = 0;
        cache_simulator->eval();
        m_trace->dump(2*curr_cycle+1);

        curr_cycle++;

        while((cache_simulator->out_icache_stall || cache_simulator->out_dcache_stall))
        {

            ////////// Feed input
            if (cache_simulator->out_icache_stall)
            {
                cache_simulator->in_icache_pc_addr       = trace_inst->pc;
                cache_simulator->in_icache_valid_pc_addr = 1;
                trace_inst->fetch_stall_cycles++;
            }
            else
            {
                cache_simulator->in_icache_valid_pc_addr = 0;
            }

            if (cache_simulator->out_dcache_stall)
            {
                cache_simulator->in_dcache_mem_read  = in_dcache_mem_read;
                cache_simulator->in_dcache_mem_write = in_dcache_mem_write;
                for (int cur_t = 0; cur_t < a.getNThds(); cur_t++)
                {
                    cache_simulator->in_dcache_in_valid[cur_t]   = in_dcache_in_valid[cur_t];
                    cache_simulator->in_dcache_in_address[cur_t] = in_dcache_in_address[cur_t];
                }
                trace_inst->mem_stall_cycles++;
            }
            else
            {
                cache_simulator->in_dcache_mem_read  = NO_MEM_READ;
                cache_simulator->in_dcache_mem_write = NO_MEM_WRITE;
                for (int cur_t = 0; cur_t < a.getNThds(); cur_t++)
                {
                    cache_simulator->in_dcache_in_valid[cur_t]   = 0;
                }
            }

            cache_simulator->clk = 1;
            cache_simulator->eval();
            m_trace->dump(2*curr_cycle);

            //////// Feed input
            if (cache_simulator->out_icache_stall)
            {
                cache_simulator->in_icache_pc_addr       = trace_inst->pc;
                cache_simulator->in_icache_valid_pc_addr = 1;
            }
            else
            {
                cache_simulator->in_icache_valid_pc_addr = 0;
            }

            if (cache_simulator->out_dcache_stall)
            {
                cache_simulator->in_dcache_mem_read  = in_dcache_mem_read;
                cache_simulator->in_dcache_mem_write = in_dcache_mem_write;
                for (int cur_t = 0; cur_t < a.getNThds(); cur_t++)
                {
                    cache_simulator->in_dcache_in_valid[cur_t]   = in_dcache_in_valid[cur_t];
                    cache_simulator->in_dcache_in_address[cur_t] = in_dcache_in_address[cur_t];
                }
            }
            else
            {
                cache_simulator->in_dcache_mem_read  = NO_MEM_READ;
                cache_simulator->in_dcache_mem_write = NO_MEM_WRITE;
                for (int cur_t = 0; cur_t < a.getNThds(); cur_t++)
                {
                    cache_simulator->in_dcache_in_valid[cur_t]   = 0;
                }
            }

            cache_simulator->clk = 0;
            cache_simulator->eval();
            m_trace->dump(2*curr_cycle+1);

            
            curr_cycle++;
            
        }

    }
}

void Core::warpScheduler()
{
    int numSteps = 0;
    bool cont;

    do
    {
        numSteps++;
        schedule_w = (schedule_w+1) % w.size();

        bool has_active_threads = (w[schedule_w].activeThreads > 0);
        bool stalled        = stallWarp[schedule_w];

        cont = ((!has_active_threads) || (stalled)) && (numSteps <= w.size());

        // cout << "&&&&&&&WID: " << schedule_w << '\n';
        // cout << "activeThreads: " << w[schedule_w].activeThreads << "\t!has_active_threads: " << (!has_active_threads) << '\n'; 

        // cout << "stalled: "  << stalled << '\n';
        // cout << "numSteps: " << numSteps << " CONT: " << cont << '\n';

    } while (cont);

    if (numSteps > w.size())
    {
        this->foundSchedule = false;
    }
    else
    {
        this->foundSchedule = true;
    }
    
}

void Core::fetch()
{

  #ifdef PRINT_ACTIVE_THREADS
  cout << endl << "Threads:";
  #endif

  // D(-1, "Found schedule: " << foundSchedule);

    if ((!inst_in_scheduler.stalled) && (inst_in_fetch.fetch_stall_cycles == 0))
    {
        // CPY_TRACE(inst_in_decode, inst_in_fetch);
        // if (w[schedule_w].activeThreads)
        {

          INIT_TRACE(inst_in_fetch);

          if (foundSchedule)
          {
              D(3, "Core step stepping warp " << schedule_w << '[' << w[schedule_w].activeThreads << ']');
              w[schedule_w].step(&inst_in_fetch);
              D(3, "Now " << w[schedule_w].activeThreads << " active threads in " << schedule_w);
            
              this->getCacheDelays(&inst_in_fetch);
              if (inst_in_fetch.stall_warp)
              {
                stallWarp[inst_in_fetch.wid] = true;
              }
          }
          warpScheduler();
        }
    }
    else
    {
        inst_in_fetch.stalled = false;
        if (inst_in_fetch.fetch_stall_cycles > 0) inst_in_fetch.fetch_stall_cycles--;
    }

    printTrace(&inst_in_fetch, "Fetch");
   
    // #ifdef PRINT_ACTIVE_THREADS
    // for (unsigned j = 0; j < w[schedule_w].tmask.size(); ++j) {
    //   if (w[schedule_w].activeThreads > j && w[schedule_w].tmask[j]) cout << " 1";
    //   else cout << " 0";
    //   if (j != w[schedule_w].tmask.size()-1 || schedule_w != w.size()-1) cout << ',';
    // }
    // #endif

  

  #ifdef PRINT_ACTIVE_THREADS
  cout << endl;
  #endif
}

void Core::decode()
{



    if ((inst_in_fetch.fetch_stall_cycles == 0) && !inst_in_scheduler.stalled)
    {
        CPY_TRACE(inst_in_decode, inst_in_fetch);
        INIT_TRACE(inst_in_fetch);
    }

    printTrace(&inst_in_decode, "Decode");
}

void Core::scheduler()
{

    if (!inst_in_scheduler.stalled)
    {
        CPY_TRACE(inst_in_scheduler, inst_in_decode);
        INIT_TRACE(inst_in_decode);
    }

    printTrace(&inst_in_scheduler, "scheduler");
}

void Core::load_store()
{
    bool do_nothing = false;
    if ((inst_in_lsu.mem_stall_cycles > 0) || (inst_in_lsu.stalled))
    {
        // LSU currently busy
        if ((inst_in_scheduler.is_lw || inst_in_scheduler.is_sw))
        {
            inst_in_scheduler.stalled = true;
        }
        do_nothing = true;
    }
    else
    {
        // LSU not busy
        if (inst_in_scheduler.is_lw || inst_in_scheduler.is_sw)
        {
            // Scheduler has LSU inst
            bool scheduler_srcs_ready = true;
            if (inst_in_scheduler.rs1 > 0)
            {
                scheduler_srcs_ready = scheduler_srcs_ready && renameTable[inst_in_scheduler.wid][inst_in_scheduler.rs1];
            }

            if (inst_in_scheduler.rs2 > 0)
            {
                scheduler_srcs_ready = scheduler_srcs_ready && renameTable[inst_in_scheduler.wid][inst_in_scheduler.rs2];
            }

            if (scheduler_srcs_ready)
            {
                if (inst_in_scheduler.rd != -1) renameTable[inst_in_scheduler.wid][inst_in_scheduler.rd] = false;
                CPY_TRACE(inst_in_lsu, inst_in_scheduler);
                INIT_TRACE(inst_in_scheduler);
            }
            else
            {
                inst_in_scheduler.stalled = true;
                // INIT_TRACE(inst_in_lsu);
                do_nothing = true;
            }
        }
        else
        {
            // INIT_TRACE(inst_in_lsu);
            do_nothing = true;
        }
    }

    if (inst_in_lsu.mem_stall_cycles > 0) inst_in_lsu.mem_stall_cycles--;

    printTrace(&inst_in_lsu, "LSU");
}

void Core::execute_unit()
{
    // cout << "$$$$$$$$$$$$$$$$$$$ EXE START\n";
    bool do_nothing = false;
    // EXEC is always not busy
    if (inst_in_scheduler.is_lw || inst_in_scheduler.is_sw)
    {
        // Not an execute instruction
        // INIT_TRACE(inst_in_exe);
        do_nothing = true;
    }
    else
    {
        bool scheduler_srcs_ready = true;
        if (inst_in_scheduler.rs1 > 0)
        {
            scheduler_srcs_ready = scheduler_srcs_ready && renameTable[inst_in_scheduler.wid][inst_in_scheduler.rs1];
            // cout << "Rename RS1: " << inst_in_scheduler.rs1 << " is " << renameTable[inst_in_scheduler.wid][inst_in_scheduler.rs1] << " wid: " << inst_in_scheduler.wid << '\n';
        }

        if (inst_in_scheduler.rs2 > 0)
        {
            scheduler_srcs_ready = scheduler_srcs_ready && renameTable[inst_in_scheduler.wid][inst_in_scheduler.rs2];
            // cout << "Rename RS2: " << inst_in_scheduler.rs1 << " is " << renameTable[inst_in_scheduler.wid][inst_in_scheduler.rs2] << " wid: " << inst_in_scheduler.wid << '\n';
        }

        if (scheduler_srcs_ready)
        {
            if (inst_in_scheduler.rd != -1) {
                // cout << "rename setting rd: " << inst_in_scheduler.rd << " to not useabel wid: " << inst_in_scheduler.wid << '\n';
                renameTable[inst_in_scheduler.wid][inst_in_scheduler.rd] = false;
            }
            CPY_TRACE(inst_in_exe, inst_in_scheduler);
            INIT_TRACE(inst_in_scheduler);
        }
        else
        {
            cout << "&&&&&&&&&&&&&&&&&&&&&&&& EXECUTE SRCS NOT READY\n";
            inst_in_scheduler.stalled = true;
            // INIT_TRACE(inst_in_exe);
            do_nothing = true;
        }
    }

    // if (!do_nothing)
    // {

    // }

    printTrace(&inst_in_exe, "execute_unit");
    // INIT_TRACE(inst_in_exe);
}

void Core::writeback()
{


    if (inst_in_wb.rd > 0) renameTable[inst_in_wb.wid][inst_in_wb.rd] = true;

    if (inst_in_wb.stall_warp)
    {
        // stallWarp[inst_in_wb.wid] = false;
        release_warp = true;
        release_warp_num = inst_in_wb.wid;
    } 


    INIT_TRACE(inst_in_wb);

    bool serviced_exe = false;
    bool serviced_mem = false;
    if ((inst_in_exe.rd > 0) || (inst_in_exe.stall_warp))
    {
        CPY_TRACE(inst_in_wb, inst_in_exe);
        INIT_TRACE(inst_in_exe);

        serviced_exe = true;
        // cout << "WRITEBACK SERVICED EXE\n";
    }

    if (inst_in_lsu.is_sw)
    {
      INIT_TRACE(inst_in_lsu);
    }
    else
    {
      if ((inst_in_lsu.rd > 0) && (inst_in_lsu.mem_stall_cycles == 0))
      {
          if (serviced_exe)
          {
              cout << "$$$$$$$$$$$$$$$$$$$$ Stalling LSU because EXE is being used\n";
              inst_in_lsu.stalled = true;
          }
          else
          {
              serviced_mem = true;
              CPY_TRACE(inst_in_wb, inst_in_lsu);
              INIT_TRACE(inst_in_lsu);

          }
      }
    }

    // if (!serviced_exe && !serviced_mem) INIT_TRACE(inst_in_wb);

    printTrace(&inst_in_wb, "Writeback");

}


bool Core::running() const {
  bool stages_have_valid = inst_in_fetch.valid_inst || inst_in_decode.valid_inst || inst_in_scheduler.valid_inst ||
                           inst_in_lsu.valid_inst   || inst_in_exe.valid_inst    || inst_in_wb.valid_inst;

  if (stages_have_valid) return true;

  for (unsigned i = 0; i < w.size(); ++i)
    if (w[i].running())
    {
        cout << "Warp ID " << i << " is running\n";
        return true;
    }
  return false;
}

void Core::printStats() const {
  unsigned long insts = 0;
  for (unsigned i = 0; i < w.size(); ++i)
    insts += w[i].insts;

  cerr << "Total steps: " << steps << endl;
  for (unsigned i = 0; i < w.size(); ++i) {
    cout << "=== Warp " << i << " ===" << endl;
    w[i].printStats();
  }
}

Warp::Warp(Core *c, Word id) : 
  core(c), pc(0x80000000), interruptEnable(true),
  supervisorMode(true), activeThreads(0), reg(0), pred(0),
  shadowReg(core->a.getNRegs()), shadowPReg(core->a.getNPRegs()), id(id),
  spawned(false), steps(0), insts(0), loads(0), stores(0)
{
  D(3, "Creating a new thread with PC: " << hex << this->pc << '\n');
  /* Build the register file. */
  Word regNum(0);
  for (Word j = 0; j < core->a.getNThds(); ++j) {
    reg.push_back(vector<Reg<Word> >(0));
    for (Word i = 0; i < core->a.getNRegs(); ++i) {
      reg[j].push_back(Reg<Word>(id, regNum++));
    }

    pred.push_back(vector<Reg<bool> >(0));
    for (Word i = 0; i < core->a.getNPRegs(); ++i) {
      pred[j].push_back(Reg<bool>(id, regNum++));
    }

    bool act = false;
    if (j == 0) act = true;
    tmask.push_back(act);
    shadowTmask.push_back(act);
  }

  Word csrNum(0);
  for (Word i = 0; i < (1<<12); i++)
  {
    csr.push_back(Reg<uint16_t>(id, regNum++));
  }

  /* Set initial register contents. */
  reg[0][0] = (core->a.getNThds()<<(core->a.getWordSize()*8 / 2)) | id;
}

void Warp::step(trace_inst_t * trace_inst) {
  Size fetchPos(0), decPos, wordSize(core->a.getWordSize());
  vector<Byte> fetchBuffer(wordSize);

  if (activeThreads == 0) return;

  // ++steps;
  
  D(3, "in step pc=0x" << hex << pc);
  cout << "help: in PC: " << hex << pc << dec << '\n';

  // std::cout << "pc: " << hex << pc << "\n";

  trace_inst->pc = pc;

  /* Fetch and decode. */
  if (wordSize < sizeof(pc)) pc &= ((1ll<<(wordSize*8))-1);
  Instruction *inst;
  bool fetchMore;

  fetchMore = false;
  unsigned fetchSize(wordSize - (pc+fetchPos)%wordSize);
  fetchBuffer.resize(fetchPos + fetchSize);
  Word fetched = core->mem.fetch(pc + fetchPos, supervisorMode);
  writeWord(fetchBuffer, fetchPos, fetchSize, fetched);
  decPos = 0;
  inst = core->iDec.decode(fetchBuffer, decPos, trace_inst);

  D(3, "Fetched at 0x" << hex << pc);
  D(3, "0x" << hex << pc << ": " << *inst);

  // Update pc
  pc += decPos;

  // Execute

  inst->executeOn(*this, trace_inst);

 
  // At Debug Level 3, print debug info after each instruction.
  #ifdef USE_DEBUG
    if (USE_DEBUG >= 3) {
      D(3, "Register state:");
      for (unsigned i = 0; i < reg[0].size(); ++i) {
        D_RAW("  %r" << setfill(' ') << setw(2) << dec << i << ':');
        for (unsigned j = 0; j < reg.size(); ++j) 
          D_RAW(' ' << setfill('0') << setw(8) << hex << reg[j][i] << setfill(' ') << ' ');
        D_RAW('(' << shadowReg[i] << ')' << endl);
      }


      D(3, "Thread mask:");
      D_RAW("  ");
      for (unsigned i = 0; i < tmask.size(); ++i) D_RAW(tmask[i] << ' ');
      D_RAW(endl);
      D_RAW(endl);
      D_RAW(endl);
    }
  #endif

  // Clean up.
  delete inst;
}

bool Warp::interrupt(Word r0) {
  if (!interruptEnable) return false;

#ifdef EMU_INSTRUMENTATION
  Harp::OSDomain::osDomain->do_int(0, r0);
#endif

  shadowActiveThreads = activeThreads;
  shadowTmask = tmask;
  shadowInterruptEnable = interruptEnable; /* For traps. */
  shadowSupervisorMode = supervisorMode;
  
  for (Word i = 0; i < reg[0].size(); ++i) shadowReg[i] = reg[0][i];
  for (Word i = 0; i < pred[0].size(); ++i) shadowPReg[i] = pred[0][i];
  for (Word i = 0; i < reg.size(); ++i) tmask[i] = 1;

  shadowPc = pc;
  activeThreads = 1;
  interruptEnable = false;
  supervisorMode = true;
  reg[0][0] = r0;
  pc = core->interruptEntry;

  return true;
}

void Warp::printStats() const {
  cout << "Steps : " << steps << endl
       << "Insts : " << insts << endl
       << "Loads : " << loads << endl
       << "Stores: " << stores << endl;

  unsigned const grade = reg[0][28];

  if (grade == 1) cout << "GRADE: PASSED\n";
  else              cout << "GRADE: FAILED "  << (grade >> 1) << "\n";
}
