# SimX LSU — per-tick draining design plan

**Goal:** lift the "one batch per tick" and "one response per tick" caps in
`LsuUnit` (and analogous places like `MemCoalescer`) **without** turning
`process_request` / `process_response` into branchy loops or sprinkling
`while`-budgets through the request handler. The two helpers should stay
single-step and trivially readable; the per-tick rate becomes a
caller-side concern.

## 1. Why this needs an abstraction, not a loop

The naive "wrap the body in a `while`" approach has three problems:

1. **Loop invariants explode.** Every early-return in `process_request`
   has to be reconsidered: does it mean "no more progress this tick" or
   "this batch is blocked, try the next request"? Today's helper
   conflates them.
2. **Per-tick budget is workload-dependent.** Response budget should
   reflect memory-side burstiness; request budget should reflect dispatch
   pressure. Hardcoding constants in the helper bodies makes them less
   reusable for other unit types (DCache, ICache).
3. **Symmetry across LSU + Coalescer.** Both `LsuUnit` and `MemCoalescer`
   have the same shape: drain an input channel, do per-item work, send to
   an output channel. The duplication will keep growing as more SimX
   units adopt this pattern.

So we want an abstraction that:
- Keeps the per-item handler **single-step, single-purpose** (a function
  that processes one item, end of story).
- Exposes the per-tick rate as a tick-level orchestration parameter.
- Is reusable across units with the same shape.

## 2. Core abstraction: `Stage`

A `Stage` is a per-tick driver that repeatedly calls a single-step
handler until either (a) the handler signals "blocked" (returns `false`),
(b) the source is empty, or (c) the per-tick budget is exhausted.

```cpp
template <typename Src, typename Step>
class Stage {
public:
    Stage(Src& source, Step step, uint32_t budget)
      : source_(source), step_(std::move(step)), budget_(budget) {}

    // Returns the number of items processed this tick (for perf/debug).
    uint32_t run() {
        uint32_t done = 0;
        while (done < budget_ && !source_empty()) {
            if (!step_())   // single-step handler decides whether to
                break;      // pop the source itself; returns false if
            ++done;         // blocked downstream and should retry next tick
        }
        return done;
    }

private:
    bool source_empty() const;   // specialized per source type

    Src&   source_;
    Step   step_;
    uint32_t budget_;
};
```

- `Src` is duck-typed: `SimChannel<T>`, `RingQueue<T>`, or anything with
  `.empty()`. `Stage` calls a free function `is_empty(source_)` so we
  don't lock in a method name.
- `Step` is a callable that returns `bool` — `true` = made progress,
  `false` = blocked (e.g., output full, allocator full); the stage halts
  for this tick.
- `Step` decides when to `pop()` the source. `Stage` never touches the
  source directly except for the empty check. This keeps the handler in
  full control of "did I consume the head?" semantics, which is exactly
  the open question that made naive looping hard.

## 3. LSU on_tick after refactor

```cpp
void LsuUnit::on_tick() {
    for (uint32_t b = 0; b < NUM_LSU_BLOCKS; ++b) {
        // Stage 1: drain memory responses (RTL: 1/cycle pipelined; budget = LSUQ_IN_SIZE)
        Stage rsp_stage(
            core_->lmem_switch(b)->RspOut,
            [this, b]() { return process_response_step(b); },
            kRspBudget);
        rsp_stage.run();

        // Stage 2: ingest from Inputs[b] into req_queue (decoupling — no budget)
        ingest_inputs(b);

        // Stage 3: dispatch batches from req_queue (RTL: 1 batch/cycle; budget = LSUQ_IN_SIZE)
        Stage req_stage(
            states_[b].req_queue,
            [this, b]() { return process_request_step(b); },
            kReqBudget);
        req_stage.run();
    }
}
```

Three orthogonal stages, each with one-line invocation. Anyone reading
`on_tick` sees the order (response → ingest → dispatch) and the budgets
in one place.

## 4. The single-step handlers — what they look like

The point of the abstraction is that these stay **simple and linear**.

### `process_response_step(b) -> bool`

```cpp
bool LsuUnit::process_response_step(uint32_t b) {
    auto& chan = core_->lmem_switch(b)->RspOut;
    if (chan.empty()) return false;
    auto& rsp = chan.peek();
    auto& entry = states_[b].pending_rd_reqs.at(rsp.tag);
    if (Outputs[b].full() && entry.eop && entry.count == rsp.mask.count())
        return false;   // would-be terminal response can't go forward yet

    /* (existing per-lane format + writeback code, unchanged) */

    if (entry.count == 0 && entry.eop)
        Outputs[b].send(entry.trace, 1);
    if (entry.count == 0)
        states_[b].pending_rd_reqs.release(rsp.tag);
    chan.pop();
    return true;
}
```

No loop, no budget arithmetic. The Stage decides how many times to call.

### `process_request_step(b) -> bool`

```cpp
bool LsuUnit::process_request_step(uint32_t b) {
    auto& state = states_[b];

    // fence drain (mirrors RTL fence_lock; same as today)
    if (state.fence_lock) {
        if (!state.pending_rd_reqs.empty()) return false;
        if (!Outputs[b].try_send(state.fence_trace)) return false;
        state.fence_lock = false;
        return true;   // made progress (fence retired); next call may dispatch
    }

    if (state.req_queue.empty()) return false;
    auto trace = state.req_queue.front();

    /* (existing fence detect / pending_rd_reqs full / AGU / batch dispatch code) */

    if (state.remain_addrs == 0) {
        if (direct_commit) Outputs[b].send(trace);
        state.req_queue.pop();
    }
    return true;
}
```

Each call dispatches one batch. The Stage drives multiple calls per tick
when budget permits.

### `ingest_inputs(b)`

Trivial loop, no budget — analogous to RTL's `core_req_ready` accepting
a request every cycle the queue isn't full:

```cpp
void LsuUnit::ingest_inputs(uint32_t b) {
    auto& state = states_[b];
    while (!Inputs[b].empty() && !state.req_queue.full()) {
        // Hold a fence at the input head until req_queue is empty
        // (mirrors the RTL fence_lock barrier).
        auto t = Inputs[b].peek();
        if (std::get<LsuType>(t->op_type) == LsuType::FENCE
            && !state.req_queue.empty())
            break;
        state.req_queue.push(t);
        Inputs[b].pop();
    }
}
```

## 5. Budget choices

The key question for fidelity: how many items per tick is "right"?

| Stage | RTL reference rate | Recommended SimX budget | Rationale |
|---|---|---|---|
| Response | 1 vector response per cycle from memory | `kRspBudget = LSUQ_IN_SIZE` (8) | Lets SimX absorb backlog if memory bursts; cap at queue size (impossible to have more outstanding than that) |
| Batch dispatch | 1 batch per cycle | `kReqBudget = LSUQ_IN_SIZE` (8) | Same — bounded by outstanding cap |
| Ingest | 1 per cycle (`core_req_ready`) | unbounded | Dispatcher-side decoupling, like RTL's `req_queue` accepting every cycle |

**Why `LSUQ_IN_SIZE` and not 1?** RTL is pipelined: while one response
is being formatted in cycle N, the next is being received in cycle N+1
and a third is being demuxed in cycle N+2. A tick-based simulator either
collapses the pipeline (drain N per tick) or models each stage as a
1-cycle delay channel. The former is simpler and accurate to first order
when the workload hasn't accumulated more than a queue's worth of items.
The latter is more faithful but requires per-stage delay channels —
overkill for the LSU's role as a thin shim.

`LSUQ_IN_SIZE` is also a natural cap: more in-flight than that is
structurally impossible.

## 6. Reuse: `MemCoalescer`

The same Stage abstraction applies to `MemCoalescer::on_tick()`:

```cpp
void MemCoalescer::on_tick() {
    Stage(RspIn, [this](){ return process_rsp_step(); }, kRspBudget).run();
    Stage(ReqIn, [this](){ return process_req_step(); }, kReqBudget).run();
}
```

Where `process_rsp_step()` and `process_req_step()` are single-step
helpers carved out of today's `on_tick`. No code duplication of the
budget-loop pattern.

The `LsuMemAdapter` and `local_mem_switch` units would benefit from the
same factoring; that's a follow-up.

## 7. What stays out of scope

- **Per-stage cycle delay channels.** Tempting (gives perfect pipeline
  fidelity) but requires every Stage's output to go through a small
  delay queue. Adds simulator overhead for marginal accuracy gain. Defer
  unless a measured workload justifies it.
- **Cross-block budget sharing.** Each LSU block runs independently; no
  shared bandwidth. Mirrors RTL where each `VX_lsu_slice` has its own
  scheduler.
- **Coroutine-based stages.** Cleaner-looking on paper but C++20
  coroutines have setup cost and a heap allocation per resume; not worth
  it for a tick-based simulator's hot path.

## 8. Migration plan

| Step | Description | Risk |
|---|---|---|
| 1 | Add `Stage<Src, Step>` to `sim/common/` | low — pure helper |
| 2 | Refactor `LsuUnit` per §3-§4 above; existing `process_request` / `process_response` split into `_step` helpers + `ingest_inputs` | low — same logic, regrouped |
| 3 | Set `kRspBudget = kReqBudget = 1` initially; verify smoke tests pass with current per-tick rate (no fidelity change yet) | low — confirms refactor correctness |
| 4 | Bump budgets to `LSUQ_IN_SIZE`; rerun benchmarks; expect IPC change on memory-bound workloads | medium — actual modeling change |
| 5 | Apply the same pattern to `MemCoalescer` (per §6) | low — same shape |
| 6 | Roll out to other tick-based shim units (LsuMemAdapter, local_mem_switch) as opportunity arises | low |

Step 3's purpose: prove the refactor preserves behavior. Step 4 is where
the actual MLP improvement lands; budget bump is a one-line config
change.

## 9. Code-complexity comparison

**Today** (per LSU block per tick):
- `process_response`: 70 lines, one peek/pop, mixes channel-management
  with response-formatting code.
- `process_request`: 130 lines, multiple early returns, no clear stage
  boundary, mixes Inputs draining with req_queue dispatch.

**After refactor**:
- `on_tick`: ~10 lines, three Stage invocations.
- `process_response_step`: ~50 lines, pure response-formatting.
- `process_request_step`: ~100 lines, pure batch-dispatch (the savings
  are mostly from extracting `ingest_inputs` and the fence drain).
- `ingest_inputs`: ~10 lines.
- `Stage`: ~20 lines.

Net effect: each function does **one thing**, and the per-tick policy
(how many times to call each one) is a property of `on_tick`, not buried
in helper logic.

## 10. Performance budget

Stage's overhead per call is a `bool` return + a counter increment. For
budget = 8 and typical workloads (1-2 items per tick), the loop runs
≤ 2 times per tick, terminating on the empty/blocked condition. Not a
hot path concern.
