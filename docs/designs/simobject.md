# SimObject Framework

The `simobject.h` framework is the core simulation runtime used by the SimX
cycle-accurate simulator. It provides four primitives:

- **`SimObject<Impl>`** — a CRTP base for cycle-tickable simulation modules.
- **`SimChannel<Pkt>`** — a typed data-plane transport between modules with
  delay-based delivery, capacity backpressure, and bind-time type conversion.
- **`SimEventLink<Msg>`** — a one-way, typed control-plane link that invokes
  a bound member-function handler in the receiver's execution context with a
  registered ≥ 1 cycle latency (see §4).
- **`SimPlatform`** — a singleton that owns objects, drives the global tick
  loop, and runs an event-driven scheduler (timing wheel + delta cycles)
  across one or more execution domains (see §10).

A working SimX module is a class derived from `SimObject<Self>` that owns
its `SimChannel`s / `SimEventLink`s as members and implements `on_tick()` /
`on_reset()`.

Two complete kernels implement this one API: the **timed** kernel
(`simobject.h` itself, cycle-accurate, the default) and the **functional**
kernel (`sim/common/simobject_functional.h`, selected by defining
`SIMX_FUNCTIONAL` in the build's `CONFIGS`; see §11). SimX picks between
them at the top of `sim/simx/types.h`; mixing objects built against different
kernels fails at link time.

---

## The Cardinal Rule

**Modules communicate *only* through channels and event links.**

A `SimObject` may observe or mutate another module's state *only* through its
bound `SimChannel` ports (`MemReq`/`MemRsp`, `result_if`, and the like) or by
sending on a bound `SimEventLink`. It must
**never reach across the ownership hierarchy** to touch another object directly:

```cpp
// WRONG — a leaf unit climbs Core -> Processor to grab the global Memory
// and read/write its DRAM backing store, bypassing the modeled cache path.
auto* gmem = core_->processor()->memsim();
gmem->write_bytes(&e.data, e.addr, e.size);

// RIGHT — the unit drives its own output channel; the request flows through
// the coalescer/cache/NoC just as the wires do in RTL.
out_req.try_send(MemReq{ .addr = e.addr, .op = MemOp::STORE, ... });
```

Why this is non-negotiable:

- **Channels *are* the wires.** The `SimChannel` graph is the SimX model of the
  chip's actual connectivity. A module's only path to the rest of the system is
  the set of ports it was wired to. Reaching around them models hardware that
  doesn't exist.
- **It preserves timing/functional fidelity and SimX↔RTL parity.** A unit that
  side-doors the backing store can read a value that, on real silicon, is still
  in flight through the cache hierarchy — producing SimX-only results the RTL
  never yields. Going through the channel path keeps the timing model and the
  functional effect consistent, which is what makes SimX a faithful oracle for RTL.
- **Hierarchy is ownership, not a call graph.** `Core` *owns* its units and
  `Processor` *owns* the `Memory`; that parent→child ownership exists for
  lifetime/construction, and must not be walked upward (`child->parent()->…`)
  or laterally to invoke a sibling's internals.

---

## 1. Tick loop

`SimPlatform::tick()` advances simulation time by one cycle:

```
fire_immediate_events()                     // delta cycle 0 settle
for each awake object, in creation order:   // the domain's ready set
  object->do_tick()                         // calls Impl::on_tick()
  fire_immediate_events()                   // settle delta cycles produced
                                            //   by this tick
++cycles_
fire registered events scheduled for cycles_ // packet deliveries with delay > 0
```

Host loops that tick until a condition holds use
`SimPlatform::run_until(stop)`, which equals `do { tick(); } while (!stop());`
cycle for cycle. Under the multi-threaded executor it costs one barrier per
cycle instead of `tick()`'s two (see §10).
`run_until(stop, voters, busy)` lets objects vote that the run continues,
which replaces the barrier with neighbor synchronization (see §10).

Two consequences:

- **Tick order matters.** Objects are ticked in the order they were created
  (`create_object<Impl>()` calls). A tick reads its inputs that were
  delivered *up to and including this cycle*; downstream consumers see its
  outputs only on the *next* cycle (when the channel events fire).
- **Delta cycles are zero-time.** Sending with `delay=0` schedules an
  immediate event that fires in the *same cycle*, between ticks. Used
  sparingly — typically for combinational fan-out like a converter or
  bypass. Default `delay=1` is a registered event.

---

## 2. SimObject\<Impl\>

```cpp
class MyUnit : public SimObject<MyUnit> {
public:
  using Ptr = std::shared_ptr<MyUnit>;
  SimChannel<MyPkt> in;
  SimChannel<MyPkt> out;

  MyUnit(const SimContext& ctx, const char* name)
    : SimObject<MyUnit>(ctx, name)
    , in(this), out(this) {}

protected:                  // ← lifecycle hooks MUST be protected
  void on_reset();
  void on_tick();

  friend class SimObject<MyUnit>;  // grant CRTP base access
};
```

Construct via `MyUnit::Create(args...)` or
`SimPlatform::instance().create_object<MyUnit>(args...)`. Both forward to
`std::make_shared<MyUnit>(SimContext{}, args...)` and register the object.

### Protected lifecycle hooks

`on_tick()` and `on_reset()` **must be protected**.
`create_object<Impl>` `static_assert`s on this — public hooks are rejected
at compile time. Only `SimPlatform` (via `do_tick()` / `do_reset()`) and
the derived class itself may invoke them. A `friend class SimObject<Self>`
is required so the framework can resolve member-pointer comparisons across
the access boundary.

### Auto-skip for passive SimObjects

`create_object<Impl>` detects whether `Impl` overrides `on_tick` /
`on_reset` by comparing `&Impl::on_tick` against
`&SimObject<Impl>::on_tick`. If equal (i.e. the default no-op is
inherited), the object is **not** added to `active_tick_` /
`active_reset_` and pays zero per-cycle cost.

So a SimObject that exists only to own a few channels — a pure
plumbing/facade — costs nothing in the hot loop. Don't define an empty
`on_tick()`; either delete the override entirely or accept the no-op
default.

Multi-level CRTP (`Derived → Intermediate → SimObject<Intermediate>`)
disables this optimization: such derivatives are conservatively kept
active.

### Sleeping between work (tick gating)

An object whose `on_tick()` would do nothing in its current state calls
`tick_sleep()` and drops out of the cycle loop; the cycle loop visits only the
domain's ready set, in creation order. What wakes it:

- a packet sent toward one of its own endpoint channels (at `send`, before
  delivery, so a skipped tick never hides a packet);
- a packet sent toward an endpoint it registered with
  `channel.wake_on_push(this)`: for an object that reads a channel another
  object owns, such as a sub-object's output;
- a pop from an endpoint it registered with `channel.wake_on_pop(this)`: for
  a producer that sleeps while its output is full;
- its own `tick_wake()`, from any other entry point (a credit return, a
  flush request, `start()`);
- a reset, which wakes every object.

A wake by an object ahead of it in creation order lets it tick the same
cycle; a wake by one behind it, the next cycle: the same as a flag tested in
a scan of every object. The contract is exactness: sleep only at a fixed
point, where every skipped tick would have been a no-op until one of the
wakes above. In particular:

- **State that changes with time** keeps the object awake: a countdown, a
  per-cycle counter that adds a non-zero value (a latency sum of outstanding
  loads), or a ready time it waits on.
- **Use `size()`, not `empty()`,** to test an input: `size()` counts packets
  still on their way in, whose send already woke the object.
- **A producer blocked on a full output** can sleep only on its true
  occupancy (`size() == capacity()`) with a pop waker; under a registered
  ready a slot freed this cycle is not offered until the next, and no event
  marks that.
- **A wake binding** goes on the endpoint itself, after elaboration bound the
  chain (in a constructor after the binds, or in `on_reset()`), and the waker
  shares the endpoint's execution domain.

Every sleep predicate must leave the cycle counts and every counter
identical; the model-parity and perf-gate cases check it.

---

## 3. SimChannel\<Pkt\>

```cpp
SimChannel<Pkt> ch(owner, capacity = 2);
```

`Pkt` must be copy-constructible. The owner is the `SimObjectBase*` that
holds the channel — used for topology introspection (`module()`,
`source()`, `sink()`).

### Endpoint vs forwarding mode

A channel is either:

- **Endpoint** — has internal storage (`RingQueue<Pkt>`). Producers
  `send()` packets; the consumer reads with `peek()` / `pop()` (or
  `try_pop()`).
- **Forwarding** — bound to a downstream channel via `bind()`. The
  channel never queues; on event delivery it invokes the downstream's
  `receive_packet()` directly. Calling `peek()`/`pop()` on a forwarded
  channel is a runtime assertion failure.

`bind()` is one-shot; rebinding asserts. The downstream's storage and
backpressure govern the upstream `full()` query.

### Send / receive

```cpp
ch.send(pkt, delay = 1);                   // schedule delivery; asserts not-full
bool ok = ch.try_send(pkt, delay = 1);     // returns false if full
const Pkt& p = ch.peek();                  // endpoint only; asserts not-empty
ch.pop();                                  // endpoint only
bool ok2 = ch.try_pop(&out_pkt);           // endpoint only
bool e = ch.empty();   bool f = ch.full();
```

`full()` and `size()` query along the bind chain — they reflect the
**endpoint's** state, not the immediate channel.

### Backpressure model: `pending_count` vs `queue_size`

When a producer `send()`s with `delay > 0`, the packet is **in flight**
inside the event wheel — not yet in any queue. The endpoint tracks both
queued packets (already delivered, waiting for the consumer) and
in-flight packets (`pending_count_`) so that `full()` accounts for both.
Producers can't oversubscribe a small endpoint queue by issuing many
delayed packets at once; capacity is enforced **at send time** against
`(queued + pending)`.

### Type-converting bind

Three `bind()` overloads:

```cpp
ch.bind(&sink);                        // exact-type, no conversion
ch.bind(&sink, [](const Src& s){...}); // explicit converter (returns Dst)
ch.bind(&sink_of_compatible_type);     // implicit-convertible Src→Dst
```

The converter runs on the upstream side at delivery time, before
`receive_packet` on the downstream. Useful when an arbiter or adapter
needs to mangle a tag or repack fields (see `MemArbiter` / `MemCrossBar`
in `types.h`).

### `tx_callback` — bus snoop

A `tx_callback` registers a function that fires from `receive_packet()`
— i.e. **on the delivery cycle**, before the packet is forwarded to the
sink (or queued in the endpoint). The callback sees the packet and the
current cycle. It's the framework's hook for "observe traffic on this
channel and react" without inserting a new SimObject in the path.

```cpp
// Count read vs write requests passing through a memory channel:
uint64_t reads = 0, writes = 0;
ch.tx_callback([&](const MemReq& req, uint64_t /*cycles*/) {
  if (req.write) ++writes; else ++reads;
});
```

```cpp
// Log every request to the trace, with the cycle it lands:
ch.tx_callback([name = ch.module()->name()](const Pkt& p, uint64_t cy) {
  std::cout << "[" << cy << "] " << name << " <- " << p << "\n";
});
```

```cpp
// Latency profiler — pair tx_callback on the response channel with a
// timestamp captured at request issue:
req_ch.tx_callback([&](const Req& r, uint64_t cy) { issued[r.tag] = cy; });
rsp_ch.tx_callback([&](const Rsp& r, uint64_t cy) {
  histogram[cy - issued[r.tag]]++;
});
```

Use it for instrumentation, side-effect events, or "snoop the bus and
poke a peer" hookups (the way RTL bus monitors snoop a request bus to
fire an event elsewhere). Avoid using it to *transform* the packet —
that's what the converter overload of `bind()` is for.

---

## 4. SimEventLink\<Msg\>

The control-plane counterpart of `SimChannel`: a one-way, typed link for
sporadic strobes — doorbells, barrier arrive/resume, completion kicks. Both
ends are declared as members; the receiver binds a member-function handler to
its end, and the sender's end is wired to the receiver's at elaboration with
the same `bind()` idiom channels use (fan-in is allowed: multiple out-ends
may target one handler end).

```cpp
// Receiver: terminal end — handler bound in the constructor.
SimEventLink<GbarArrive> gbar_arrive_in;
...
gbar_arrive_in.bind(this, &Cluster::on_gbar_arrive);

// Sender: out end — wired at elaboration like any channel.
SimEventLink<GbarArrive> gbar_arrive_out;
...
core->gbar_arrive_out.bind(&cluster->gbar_arrive_in);

// Use, from the sender's own code:
gbar_arrive_out.send({bar_id, count, core_id});   // delay = 1 implied
```

`send(msg, delay = 1)` is fire-and-forget: it cannot fail and cannot be
refused, and the bound handler runs in the **receiving module's** execution
context at cycle `C + delay` (`delay == 0` asserts). Same-cycle deliveries
from concurrent senders are merged in a canonical order, so behavior is
identical serial and multi-threaded. A handler may mutate its own module's
state and send on its own links/channels — nothing else; messages must be
self-contained values (never smuggle a pointer to mutable state).

| | `SimEventLink` | `SimChannel` |
|---|---|---|
| Traffic | sporadic strobe | stream |
| Refusable? | no — delivery cannot fail | yes — `try_send` fails when full |
| Consumer code | none: bound handler is invoked | polls `empty()`/`peek()`/`pop()` |
| Occupancy | meaningless | queue depth + delay *are* the timing model |
| Hardware analog | a strobe/doorbell wire | a valid/ready pipe with a FIFO |

An event link is also the **only** way to trigger behavior on a module in
another execution domain (§10) — a direct method call across a domain
boundary is a data race the framework cannot see.

---

## 5. Events

### `SimChannelEvent<Pkt>` (typed)

Created by `SimChannel::send()`. On fire, calls
`channel_->receive_packet(pkt_)` which triggers any `tx_cb_`, follows the
bind chain, and lands at the endpoint's queue. Pool-allocated.

### `SimCallEvent<Pkt>` (generic)

Created by `SimPlatform::schedule(func, pkt, delay)`. On fire, calls
`func(pkt)`. Useful for arbitrary deferred work that doesn't ride a
channel — e.g. periodic counter rollover, deferred wake-ups. Held
function payload is bounded to ~48 bytes (small-function optimization
via `SmallFunction`).

### Wheel and lanes

Three storage tiers per execution domain:

- **The wheel** — `delay >= 2`: a hashed timing wheel with
  `WHEEL_SIZE = 4096` buckets indexed by `cycle & WHEEL_MASK`. Events
  scheduled for cycles past the current wraparound are revisited each
  pass. Fire on `tick()` after the per-object loop.
- **The next-cycle lane** — `delay == 1`, the common case: a flat vector per
  due-cycle parity, fired right after the wheel bucket of the same cycle. A
  domain's time only moves forward, so every longer-delay event due that
  cycle was scheduled before any delay-1 event: wheel first, then the lane,
  is exactly the order of a single wheel.
- **The immediate lane** — `delay == 0`, fires **between ticks** of the
  current cycle in schedule order, including events a firing appends.
  Multiple delta passes settle combinational chains within the same cycle.

A lane event carries a small trivially copyable packet (a pointer, a
credit) inline, with no allocation and no virtual call; a larger packet
rides a pooled `SimChannelEvent`, as on the wheel.

Use `delay=0` only for genuine combinational paths (a forwarder that
must complete in-cycle). Default to `delay=1` for normal flow.

### Inflight counter and `idle()`

Every endpoint counts the packets sent toward it but not yet consumed:
incremented on `reserve()`, decremented on queue pop. The count lives in the
endpoint's execution domain (one cache line per domain, no atomics), because
only that domain's context reserves toward or pops from its endpoints.

Host-side quiescence is a framework service: `SimPlatform::idle()` is true
when no packet is in flight **and** no deferred cross-domain delivery is
pending. Host run/flush loops must use `idle()` — reassembling the test from
framework internals is exactly the forgotten-predicate bug it exists to
prevent.

---

## 6. Common patterns

### Module owning input/output channels

```cpp
class MyFifo : public SimObject<MyFifo> {
public:
  SimChannel<Req> Inputs;
  SimChannel<Req> Outputs;

  MyFifo(const SimContext& ctx, const char* name)
    : SimObject<MyFifo>(ctx, name), Inputs(this), Outputs(this) {}

protected:
  void on_tick() {
    if (Inputs.empty() || Outputs.full()) return;
    Outputs.send(Inputs.peek());
    Inputs.pop();
  }
  friend class SimObject<MyFifo>;
};
```

### `std::array<SimChannel<Pkt>, N>` member

`SimChannel<Pkt>` is not default-constructible (it needs the owner).
Use the `make_sim_channels<Pkt, N>(this)` helper:

```cpp
std::array<SimChannel<Req>, N> Inputs = make_sim_channels<Req, N>(this);
```

For runtime-sized vectors, use `std::vector<SimChannel<Req>>(N, this)`
— the per-element constructor is the variadic forwarding form
`(owner, capacity)`.

### Pure plumbing / facade SimObject

A SimObject that only owns channels and never overrides `on_tick`:
no per-cycle cost, but still gets `name()` and topology. The `DxaUnit`
pattern (where decode happens via `process(trace)` called from
`SfuUnit::on_tick`) is a refinement: don't even make it a SimObject —
hold the channel reference and skip the framework entirely.

### Channel snooping (bus-fire side effect)

Use `tx_callback` on the upstream channel. Fires at delivery cycle, sees
the packet. No need for an intermediate snoop SimObject — this avoids
adding a SimObject (and its per-cycle tick cost) just to react to
traffic on a channel you've already wired. See §3 for examples.

### Scheduling deferred work without a channel

```cpp
SimPlatform::instance().schedule(
  [this](const State& s) { this->resume(s.wid); },
  state, /*delay=*/3);
```

Schedules an arbitrary callback to fire 3 cycles in the future. Avoid
in hot paths — `SimChannelEvent` is more efficient when the work is a
packet delivery.

---

## 7. Reset

`SimPlatform::reset()`:

1. Drains all scheduled events (registered + immediate).
2. Calls `do_reset()` on every active-reset object.
3. Resets `cycles_ = 0`.

The in-flight counts are *not* reset by the platform, and `idle()` is their
only reader. Module `on_reset()` should clear all internal
queues and counters; channels are reset as a side effect (storage is
reconstructed at construction; reset doesn't recreate them).

---

## 8. Lifecycle ownership

`SimPlatform` holds `shared_ptr<SimObjectBase>` to every created
object. Module-to-module references (e.g. one SimObject holding a
pointer to another) should use raw pointers or `weak_ptr` — never
`shared_ptr`, which would create reference cycles that block cleanup
(`SimPlatform::cleanup()` clears its vector, but cycles among modules
won't release).

`SimChannel`s are *value members* of their owning SimObject and live
exactly as long as the owner. Bindings hold raw pointers; if the
upstream/downstream is destroyed before the channel is unbound, sends
are undefined.

---

## 9. Topology introspection

`SimChannelBase::module()` / `source()` / `sink()` give the bind
topology. `SimObjectBase::name()` returns the registered name.
Together these support tools like a topology dump or a cycle tracer.

```cpp
// Walk a channel chain to its endpoint:
SimChannelBase* ep = &my_ch;
while (ep->sink()) ep = ep->sink();
// ep now points at the final endpoint channel.
```

---

## 10. Execution domains and multi-threading

The platform partitions the design into **execution domains**: topology
containers (Socket/Cluster) open a `SimPlatform::DomainScope` per partition,
and every object created inside inherits that domain. Domain 0 is the
uncore/default domain (memory system, KMU, host-facing blocks); each socket
gets its own domain. Leaf modules never manage domains themselves.

Defining `SIMX_MT=<T>` in the build's `CONFIGS` runs the domains on `T`
lockstep worker threads (default serial). The kernel headers carry no build
knobs: `sim/simx/types.h` resolves the macro once into the
`SIMX_NUM_WORKERS` constant, and the application hands it to
`SimPlatform::set_num_workers()` before the first tick.
The executor guarantees **bit-identical cycles for every thread count**: all
per-domain state (scan list, wheel, delta events) is touched only by its
owner, cross-domain deliveries are merged in a canonical `(due, src, seq)`
order, and each cycle is fenced by barriers. A contributor who follows the
Cardinal Rule gets deterministic parallelism for free.

A cycle is one barrier inside `run_until()`: the cycle-end barrier's release
step commits the cycle and runs the host's stop check while every worker is
parked, and releasing the barrier starts the next cycle. A bare `tick()`
costs two (release, then join). The barrier itself has no shared
read-modify-write: each worker announces its arrival on its own cache line
and the coordinator (thread 0) collects them. Domains go to threads
round-robin, with a last partial round placed from the highest thread down,
so the coordinator, which owns the uncore domain and runs the release steps,
takes no extra socket.

A host loop that cannot end while some objects are busy uses
`run_until(stop, voters, busy)` instead. After each cycle every voter is
evaluated with `busy(voter)` by the thread that ticks it, so model code never
names a domain; true means the run certainly continues, for example because a
core still runs. `stop(busy)` gets the OR of the votes; with `busy` set it
must return `false` and touch only the uncore domain's state, since other
domains may be mid-cycle. With workers, this replaces the per-cycle barrier
with **neighbor synchronization**:

- Each domain publishes, on its own cache line, how many cycles it has
  completed and its vote, and starts cycle `c` once every neighbor has
  completed `c` cycles. Neighbors are the domains a channel bind chain joins,
  plus the uncore with every domain. Neighbors stay within one cycle of each
  other, which keeps the triple-buffered mailboxes valid. Sockets do not wait
  for each other directly, but each waits for the uncore, so none gets more
  than two cycles ahead of the slowest.
- Thread 0 decides the stop check for the cycle ending at `c` when it starts
  the next one, once every domain has completed `c`. A thread with a busy
  domain may run ahead, and then the run cannot end at `c`; a thread whose
  domains all voted idle waits for the decision. When every vote is idle,
  every other thread is parked, so `stop(false)` sees exactly the lockstep
  run's state: the stop cycle and every counter equal the lockstep run's.
- A cross-domain call between two domains that are not neighbors aborts the
  run, because the synchronization could not order it.

`Processor::run()` lets its cores vote "running" when the processor has one
cluster and no VM: there a running core leaves the check nothing else to do.
Other host loops (cache flushes, copy-engine transfers) keep the barrier. In
the functional kernel the unit is a quantum instead of a cycle.

Cross-domain calls travel through per-(target, source) mailbox lanes, each
written by one source and drained by one target, so no lock or shared counter
sits on the path. A lane is triple-buffered by epoch (a cycle, or a quantum in
the functional kernel): a source writes the current epoch, and the target
drains the previous one after the barrier that closed it. A call therefore runs
at the first epoch boundary at or after its due cycle, in `(due, src, seq)`
order; with one-cycle epochs this is exactly the order of a single shared
inbox.

What makes a module MT-safe is not new code — it is the *absence* of illegal
code, enforced by lint and asserts:

| A component author may | A component author may not |
|---|---|
| mutate own state; `schedule()` for self | touch another module's fields or call its methods across a domain |
| `send`/`try_send` on own out-channels (delay ≥ 1 across boundaries) | read `full()`/`size()` of a foreign-domain endpoint |
| declare `SimEventLink` ends; bind handlers; send on own out-ends | use `cross_call`, `cross_pending`, `domain()`, `std::atomic`, threads, locks |
| read anything from host context between cycles | hand-roll termination predicates (use `SimPlatform::idle()`) |

A channel chain that crosses a domain boundary must pass through a
registered boundary stage (`RegSlice` in `sim/simx/regslice.h` — a real
pipeline register with credit-based backpressure, owned by the sending
domain). Topology validation runs at the first reset: an unregistered
cross-domain edge is reported and caps execution at one thread until it is
converted. Host code (between ticks) may read any module's state directly
and mutate only through component API calls (reset, start, dcr_write) — the
same rule testbenches follow.

---

## 11. Functional kernel

Building with `CONFIGS="-DSIMX_FUNCTIONAL"` (in a separate build tree such
as `build32_functional/`) selects the functional kernel:
the same executor with timing removed, for full-speed architectural runs
(conformance suites, ISA bring-up).

- Every `send`/`schedule` delay ≥ 1 collapses to one cycle; `delay == 0`
  keeps its delta semantics.
- Backpressure is disabled: `full()` is always `false`, `try_send` always
  succeeds, storage is unbounded (with a high-water debug assert).
- The MT executor, domains, canonical ordering, and topology validation are
  retained — `-DSIMX_MT` composes and cycles remain bit-identical across
  thread counts.
- `-DSIMX_QUANTUM=<Q>` (resolved in `sim/simx/types.h`, handed to
  `SimPlatform::set_quantum()`) lets domains run `Q` cycles between
  barriers; a cross-domain call runs at the first quantum boundary at or
  after its due cycle.
  Results are bit-identical for every thread count at a given `Q`, but `Q`
  itself changes the (non-physical) cycle count. A larger `Q` amortizes
  synchronization but delays every crossing — an L1 miss into the uncore
  domain waits for the boundary — so `Q` around 100 is a good start. The
  timed kernel keeps per-cycle lockstep and rejects `Q > 1`.
- Cycle counts are monotonic but **non-physical**: a functional build must
  never feed `perf_gate`, `model_parity`, or baseline regeneration.

Component code is identical under both kernels; the choice is per build
tree, and a mixed-kernel link fails at link time for kernel symbols
(`RegSlice` sits outside that guard and relies on the per-tree choice
being uniform).
