// Copyright © 2019-2023
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "wctl_unit.h"
#include <iostream>
#include "core.h"
#include "scheduler.h"
#include "constants.h"
#include "debug.h"

using namespace vortex;

#ifdef VX_CFG_DIVERGE_TYPE_NV_ITS
// Release a convergence barrier — clearing both masks so the bid is reusable —
// once every participant still able to arrive has arrived: exited threads are
// already masked out by TMC and yielded participants are excluded so a parked
// spinner cannot hold back reconvergence forever.
static void its_try_release(warp_t& warp, uint32_t b) {
  auto& participate = warp.bar_participate.at(b);
  auto& arrived = warp.bar_arrived.at(b);
  if (participate.any() && arrived.any()
   && !(participate & ~warp.yielded & ~arrived).any()) {
    participate.reset();
    arrived.reset();
  }
}
#endif


bool WctlUnit::process(instr_trace_t* trace) {
  bool release_warp = trace->fetch_stall;
  auto wctl_type    = std::get<WctlType>(trace->op_type);
  auto& sched       = core_->scheduler();
  auto& warp        = sched.warp(trace->wid);
  auto instrArgs    = trace->instr_ptr->get_args();
  auto wctlArgs     = std::get<IntrWctlArgs>(instrArgs);
  uint32_t num_threads = VX_CFG_NUM_THREADS;
  auto& rs1_data    = trace->src_data[0];
  auto& rs2_data    = trace->src_data[1];

  uint32_t thread_start = 0;
  for (; thread_start < num_threads; ++thread_start) {
    if (warp.tmask.test(thread_start)) break;
  }
  int32_t thread_last = num_threads - 1;
  for (; thread_last >= 0; --thread_last) {
    if (warp.tmask.test(thread_last)) break;
  }

  switch (wctl_type) {
  case WctlType::TMC: {
    ThreadMask next_tmask(num_threads);
    for (uint32_t t = 0; t < num_threads; ++t) {
      next_tmask.set(t, rs1_data.at(thread_last).u & (1 << t));
    }
    if (trace->eop) {
#ifdef VX_CFG_DIVERGE_TYPE_SPLIT
      // SCS: record lanes a TMC turns off as exited ONLY while the warp has
      // schedulable parked work. scs_done exists solely to stop a resuming parked
      // subgroup (e.g. a lock holder) from resurrecting a lane that already left
      // the kernel; with nothing parked there is nothing to protect. Tracking it
      // unconditionally mis-reads a plain mask-narrowing TMC (e.g. the legacy
      // vx_spawn_threads work-distribution loop narrows then re-widens the mask
      // each wave) as a permanent exit, so the re-widen is filtered to nothing and
      // only lane 0 survives. With nothing parked, TMC is honoured verbatim —
      // exactly baseline behaviour.
      bool has_parked = !warp.scs_pending.empty() || !warp.scs_runnable.empty()
                     || warp.scs_parked.any();
      if (has_parked) {
        warp.scs_done = warp.scs_done | (warp.tmask & ~next_tmask);
      }
#endif
#ifdef VX_CFG_DIVERGE_TYPE_NV_ITS
      // ITS: threads diverge on per-thread PCs, so a kernel-exit TMC reaches
      // the scheduler from whichever group gets there first — it must opt out
      // only the issuing group's threads (the patent's per-thread EXIT), never
      // the whole warp: parked or yielded threads still have work to run.
      // A non-zero TMC keeps whole-warp semantics (spawn work loops narrow and
      // re-widen the mask convergently).
      ThreadMask next_amask = next_tmask.any() ? next_tmask
                                               : (warp.amask & ~warp.tmask);
      warp.amask = next_amask;
      if (next_tmask.any()) {
        for (uint32_t t = 0; t < num_threads; ++t) {
          if (next_tmask.test(t))
            warp.tpc[t] = trace->PC + 4;
        }
      }
      // Exited threads leave every barrier, and each release condition is
      // re-evaluated — otherwise an exit that removes a parked participant
      // leaves that barrier permanently unsatisfiable (latent bug in the
      // reference implementation; the patent ignores exited threads).
      warp.yielded &= next_amask;
      for (uint32_t b = 0; b < VX_CFG_ITS_NUM_BARRIERS; ++b) {
        warp.bar_participate[b] &= next_amask;
        warp.bar_arrived[b] &= next_amask;
        its_try_release(warp, b);
      }
      if (!next_tmask.any() && next_amask.any()) {
        // Group exit with survivors: the warp stays schedulable; the next
        // group pick rebuilds tmask from the surviving threads.
        release_warp = true;
        break;
      }
#endif
      release_warp = core_->setTmask(trace->wid, next_tmask);
    }
  } break;
  case WctlType::WSPAWN: {
    if (trace->eop) {
      release_warp = core_->wspawn(rs1_data.at(thread_last).u, rs2_data.at(thread_last).u);
    }
  } break;
#ifdef VX_CFG_DIVERGE_TYPE_NV_ITS
  // ITS: the legacy IPDOM ops are architectural no-ops (divergence is handled
  // by per-thread PCs + convergence barriers). SPLIT still writes its result
  // register so IPDOM-compiled binaries remain executable.
  case WctlType::SPLIT: {
    for (uint32_t t = thread_start; t < num_threads; ++t) {
      trace->dst_data[t].i = 0;
    }
    if (trace->eop)
      release_warp = true;
  } break;
  case WctlType::JOIN: {
    if (trace->eop)
      release_warp = true;
  } break;
#else
  case WctlType::SPLIT: {
    Word next_pc = trace->PC + 4;
    ThreadMask then_tmask(num_threads);
    ThreadMask else_tmask(num_threads);
    auto not_pred = wctlArgs.is_cond_neg;
    for (uint32_t t = 0; t < num_threads; ++t) {
      auto cond = (rs1_data.at(t).i & 0x1) ^ not_pred;
      then_tmask[t] = warp.tmask.test(t) && cond;
      else_tmask[t] = warp.tmask.test(t) && !cond;
    }
    ThreadMask next_tmask = warp.tmask;
    bool is_divergent = then_tmask.any() && else_tmask.any();
    auto stack_size = warp.ipdom_stack.size();
    // stack_size captured pre-push is what each pid writes to its
    // dst_data — it's the kernel-visible value at this PC.
    for (uint32_t t = thread_start; t < num_threads; ++t) {
      trace->dst_data[t].i = stack_size;
    }
    // ipdom_stack push + tmask update
    if (trace->eop) {
      if (is_divergent) {
        if (stack_size == sched.ipdom_size()) {
          std::cout << "IPDOM stack is full! size=" << stack_size << ", PC=0x" << std::hex << warp.PC << std::dec << " (#" << trace->uuid << ")\n" << std::flush;
          std::abort();
        }
        next_tmask = (then_tmask.count() <= else_tmask.count()) ? then_tmask : else_tmask;
        warp.ipdom_stack.emplace(warp.tmask, next_pc);
        core_->perf_stats().divergence += 1;
      }
      release_warp = core_->setTmask(trace->wid, next_tmask);
    }
  } break;
  case WctlType::JOIN: {
    auto stack_ptr = rs1_data.at(thread_last).u;
    auto stack_size = warp.ipdom_stack.size();
    ThreadMask next_tmask = warp.tmask;
    // ipdom_stack pop + tmask update
    if (trace->eop) {
      if (stack_ptr != stack_size) {
        if (warp.ipdom_stack.empty()) {
          std::cout << "IPDOM stack is empty!\n" << std::flush;
          std::abort();
        }
        if (warp.ipdom_stack.top().fallthrough) {
          next_tmask = warp.ipdom_stack.top().orig_tmask;
          warp.ipdom_stack.pop();
        } else {
#ifdef VX_CFG_DIVERGE_TYPE_SPLIT
          // SCS: capture the arriving subgroup's forward (post-join) PC before
          // it is set aside, so the watchdog can resume it if the sibling spins.
          warp.ipdom_stack.top().passed_tmask = warp.tmask;
          warp.ipdom_stack.top().passed_pc    = trace->PC + 4;
          warp.ipdom_stack.top().has_passed   = true;
#endif
          next_tmask = ~warp.tmask & warp.ipdom_stack.top().orig_tmask;
          warp.PC = warp.ipdom_stack.top().else_PC;
          warp.ipdom_stack.top().fallthrough = true;
        }
      }
      release_warp = core_->setTmask(trace->wid, next_tmask);
    }
  } break;
#endif // !VX_CFG_DIVERGE_TYPE_NV_ITS
  case WctlType::BAR: {
    uint32_t arg1 = rs1_data[thread_last].u;
    uint32_t arg2 = rs2_data[thread_last].u;
    uint32_t bar_id = bar_decode_id(arg1, VX_CFG_NUM_BARRIERS);
    bool is_sync_bar = (bool)wctlArgs.is_sync_bar;
    // rs2[31] flags expect_tx semantics on the same opcode as barrier_arrive.
    // Lower bits carry the count of pending transactions to register.
    bool is_tx_expect = wctlArgs.is_bar_arrive && ((arg2 >> 31) & 0x1);
    if (wctlArgs.is_bar_arrive) {
      // BAR.ARRIVE writes back the current barrier phase (sampled before this op modifies it).
      uint32_t phase = core_->scheduler().barrier_unit().get_phase(bar_id) & 0x1u;
      for (uint32_t t = thread_start; t < num_threads; ++t) {
        if (!warp.tmask.test(t)) continue;
        trace->dst_data[t].i = phase;
      }
    }
    if (trace->eop) {
      if (is_tx_expect) {
        // Pre-register pending transaction event(s); does not advance arrival count.
        uint32_t tx_count = arg2 & 0x7fffffff;
        core_->barrier_event_attach(bar_id, tx_count);
      } else if (trace->wb || is_sync_bar) {
        core_->barrier_arrive(bar_id, arg2, trace->wid, is_sync_bar);
        if (is_sync_bar) {
          release_warp = false;
        }
      } else {
        release_warp = !core_->barrier_wait(bar_id, arg2, trace->wid);
      }
    }
  } break;
  case WctlType::PRED: {
#ifdef VX_CFG_DIVERGE_TYPE_NV_ITS
    // ITS: PRED is an architectural no-op (see SPLIT/JOIN above).
    if (trace->eop)
      release_warp = true;
#else
    ThreadMask pred(num_threads);
    auto not_pred = wctlArgs.is_cond_neg;
    for (uint32_t t = 0; t < num_threads; ++t) {
      auto cond = (rs1_data.at(t).i & 0x1) ^ not_pred;
      pred[t] = warp.tmask.test(t) && cond;
    }
    ThreadMask next_tmask = warp.tmask;
#ifdef VX_CFG_DIVERGE_TYPE_SPLIT
    bool reconverged = false;
    if (!sched.scs_enabled()) {
      next_tmask = pred.any() ? pred : ThreadMask(num_threads, rs2_data.at(thread_last).u);
      if (trace->eop) {
        release_warp = core_->setTmask(trace->wid, next_tmask);
      }
      break;
    }
    if (pred.any()) {
      next_tmask &= pred;
    } else {
      // Loop reconverged (no lane still needs to spin). Restore the loop's
      // participants from the CURRENT subgroup plus the lanes it parked-pending
      // in this loop — NOT from the rs2 (csrr-tmask) snapshot. Under SCS the
      // running subgroup is a split of the original warp, so the compiler's rs2
      // snapshot is stale and would restore the wrong lanes (dropping the
      // running lane entirely). Escaped lanes (scs_parked) are excluded in
      // setTmask, so the un-cancelled pending lanes rejoin while escaped ones
      // keep running independently.
      next_tmask = warp.tmask;
      for (auto& p : warp.scs_pending)
        next_tmask |= p.tmask;
      reconverged = true;
    }
    if (trace->eop) {
      if (reconverged) {
        // The un-cancelled pending lanes were folded into next_tmask above;
        // drop the now-restored pending records.
        warp.scs_pending.clear();
      } else {
        ThreadMask just_off = warp.tmask & ~next_tmask;
        if (just_off.any()) {
          // SCS: the lanes this vx_pred masked off (e.g. lock acquirers that won)
          // become a DISTINCT pending subgroup resuming at the next instruction —
          // the loop's back-branch, which routes this now-uniform subgroup
          // straight to the loop exit — and carrying its own reconvergence
          // snapshot. Distinct per-loop entries (vs. one shared {mask,PC} slot)
          // are what stop two different blocking loops from conflating. Kept
          // pending (cancellable) until the loop reconverges or the warp stalls.
          warp.scs_pending.emplace_back(just_off, trace->PC + 4, warp.ipdom_stack);
          sched.observe_split_contexts(warp);
        }
      }
      release_warp = core_->setTmask(trace->wid, next_tmask);
    }
#else
    if (pred.any()) {
      next_tmask &= pred;
    } else {
      next_tmask = ThreadMask(num_threads, rs2_data.at(thread_last).u);
    }
    if (trace->eop) {
      release_warp = core_->setTmask(trace->wid, next_tmask);
    }
#endif
#endif // !VX_CFG_DIVERGE_TYPE_NV_ITS
  } break;
  case WctlType::WSYNC:
    release_warp = true;
    break;
  case WctlType::YIELD: {
    // SCS: deschedule the running split, rotate to the next runnable one.
    // Decoded in every mode (threadsplit binaries carry it); acts only under
    // SPLIT, elsewhere it just unlocks the warp.
    if (trace->eop) {
#ifdef VX_CFG_DIVERGE_TYPE_SPLIT
      release_warp = sched.yield_warp(trace->wid);
#elif defined(VX_CFG_DIVERGE_TYPE_NV_ITS) && defined(VX_CFG_ITS_YIELD_ENABLE)
      // ITS: the issuing group enters the Yielded state (resume PC is already
      // tpc = PC+4); barriers whose missing participants are now all yielded
      // release so blocked threads make progress.
      warp.yielded |= warp.tmask;
      for (uint32_t b = 0; b < VX_CFG_ITS_NUM_BARRIERS; ++b) {
        its_try_release(warp, b);
      }
      release_warp = true;
#else
      release_warp = true;
#endif
    }
  } break;
#ifdef VX_CFG_DIVERGE_TYPE_NV_ITS
  case WctlType::BAR_ADD: {
    // ITS: the executing group registers as participants of barrier <bid>;
    // threads are not masked and may diverge at the next branch.
    if (trace->eop) {
      warp.bar_participate.at(wctlArgs.bid) |= warp.tmask;
      release_warp = true;
    }
  } break;
  case WctlType::BAR_WAIT: {
    // ITS: the executing group arrives at barrier <bid> and blocks (the
    // scheduler excludes arrived threads from group selection). Resume PC is
    // already tpc = PC+4 from decode. Release — clearing both masks so the bid
    // is reusable — once every participant has arrived.
    if (trace->eop) {
      auto& participate = warp.bar_participate.at(wctlArgs.bid);
      auto& arrived = warp.bar_arrived.at(wctlArgs.bid);
      // Arrival is masked to actual participants, and a wait on an empty
      // barrier passes through. Barrier ids are allocated per function, so a
      // callee's bid can collide with a live caller barrier (e.g. a divergent
      // loop inside libm sinf under the kernel's own loop barrier); unmasked
      // arrival would poison the release equality forever. The patent solves
      // this with barrier state save/restore across calls (BMOV); this
      // pass-through rule only weakens reconvergence packing, never blocks a
      // thread on a barrier it never joined.
      arrived |= (warp.tmask & participate);
      its_try_release(warp, wctlArgs.bid);
      release_warp = true;
    }
  } break;
#endif // VX_CFG_DIVERGE_TYPE_NV_ITS
  default:
    std::abort();
  }
  return release_warp;
}
