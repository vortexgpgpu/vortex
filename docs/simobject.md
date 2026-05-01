# SimObject Framework

The `simobject.h` framework is the core simulation runtime used by the SimX
cycle-accurate simulator. It provides three primitives:

- **`SimObject<Impl>`** — a CRTP base for cycle-tickable simulation modules.
- **`SimChannel<Pkt>`** — a typed transport between modules with delay-based
  delivery, optional capacity backpressure, and bind-time type conversion.
- **`SimPlatform`** — a singleton that owns objects, drives the global tick
  loop, and runs an event-driven scheduler (timing wheel + delta cycles).

A working SimX module is a class derived from `SimObject<Self>` that owns
its `SimChannel`s as members and implements `on_tick()` / `on_reset()`.

---

## 1. Tick loop

`SimPlatform::tick()` advances simulation time by one cycle:

```
fire_immediate_events()                     // delta cycle 0 settle
for each active object:
  object->do_tick()                         // calls Impl::on_tick()
  fire_immediate_events()                   // settle delta cycles produced
                                            //   by this tick
++cycles_
fire registered events scheduled for cycles_ // packet deliveries with delay > 0
```

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

## 4. Events

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

### Wheel + immediate buckets

Two storage tiers:

- **Registered events** (`reg_events_`) — a hashed timing wheel with
  `WHEEL_SIZE = 4096` buckets indexed by `cycle & WHEEL_MASK`. Events
  scheduled for cycles past the current wraparound are revisited each
  pass. Fire on `tick()` after the per-object loop.
- **Immediate events** (`imm_events_`) — `delay == 0`, fire **between
  ticks** of the current cycle in delta-order (`delta_` is bumped per
  scheduled). Multiple delta passes settle combinational chains within
  the same cycle.

Use `delay=0` only for genuine combinational paths (a forwarder that
must complete in-cycle). Default to `delay=1` for normal flow.

### Inflight counter

`SimChannelBase::inflight_count()` is a process-global counter
incremented on `reserve()` and decremented on queue pop. Useful for
deadlock detection (a tick that drops to zero traffic and stays there
when work is expected) and for end-of-simulation drain assertions.

---

## 5. Common patterns

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

## 6. Reset

`SimPlatform::reset()`:

1. Drains all scheduled events (registered + immediate).
2. Calls `do_reset()` on every active-reset object.
3. Resets `cycles_ = 0`.

Inflight-count is *not* reset by the platform — clear it externally if
your test depends on it. Module `on_reset()` should clear all internal
queues and counters; channels are reset as a side effect (storage is
reconstructed at construction; reset doesn't recreate them).

---

## 7. Lifecycle ownership

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

## 8. Topology introspection

`SimChannelBase::module()` / `source()` / `sink()` give the bind
topology. `SimObjectBase::name()` returns the registered name.
Together these support tools like a topology dump or a cycle tracer.

```cpp
// Walk a channel chain to its endpoint:
SimChannelBase* ep = &my_ch;
while (ep->sink()) ep = ep->sink();
// ep now points at the final endpoint channel.
```
