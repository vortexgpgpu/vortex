# SimX Simulation Framework

SimX models are built on one kernel header,
[`sim/common/simobject.h`](../sim/common/simobject.h), and one library
header, [`sim/common/simobject_lib.h`](../sim/common/simobject_lib.h). A
model is a set of **units** joined by **ports**: a unit is a hardware block,
a port is the wire between two of them, and the latency of a fixed pipeline
is a property of the connection. Units in namespace `vortex`:

| type | role |
|---|---|
| `SimObject<U>` | base of every model: `on_tick()` or handler ports, `busy()`, `on_reset()`, `tick_wake()` |
| `SimPortIn<T>`, `SimPortInArray<T>` | an input port, owned by the unit that reads it |
| `SimPortOut<T>`, `SimPortOutArray<T>` | an output port, owned by the unit that writes it, bound once |
| `SimEventLink<Msg>` | a one-way control strobe (doorbell, barrier arrive/resume) |
| `SimPlatform` | the singleton that owns the units and runs the cycles |

The header comments are the reference for each call; this document gives
the model and the rules. The framework's own tests are in
[`sim/common/tests/simobject/`](../sim/common/tests/simobject/):
`conformance.cpp` checks port timing against golden logs, `features.cpp`
covers `busy()`, multi-cycle units and the functional kernel, and
`library.cpp` checks the library units against golden logs.

| § | contents |
|---|---|
| [–](#the-cardinal-rule) | The cardinal rule |
| [1](#1-units) | Units: `on_tick()`, handler ports, `busy()`, `on_reset()`, `tick_wake()` |
| [2](#2-ports) | Ports: `SimPortIn`, `SimPortOut`, bind, internal registers, wires, credits |
| [3](#3-timing-rules) | Timing rules and how a unit is woken |
| [4](#4-event-links) | Event links |
| [5](#5-library-units) | Library units |
| [6](#6-containers) | Containers and child ports |
| [7](#7-execution-domains-and-multi-threading) | Execution domains, multi-threading, host loops |
| [8](#8-functional-kernel) | The functional kernel |

---

## The Cardinal Rule

**Units communicate *only* through their ports and event links.**

A unit observes or changes another unit's state *only* through its own
`SimPortIn`/`SimPortOut` ports (`MemReq`/`MemRsp`, results, credits) or by sending on a
bound `SimEventLink`. It **never reaches across the ownership hierarchy** to
touch another unit directly:

```cpp
// WRONG: a leaf unit climbs Core -> Processor to the global Memory and
// writes the DRAM image, bypassing the modeled cache path.
core_->processor()->memsim()->write_bytes(&e.data, e.addr, e.size);

// RIGHT: the unit drives its own output; the request flows through the
// coalescer, caches and NoC, as the wires do in RTL.
if (!mem_req_out.full()) {
  mem_req_out.send(MemReq{.addr = e.addr, .op = MemOp::STORE, ...});
}
```

- **Ports are the wires.** The port graph is the model of the chip's
  connectivity; a unit's only path to the rest of the system is the ports it
  was bound to. Reaching around them models hardware that does not exist.
- **It keeps SimX an RTL oracle.** A unit that side-doors the backing store
  reads values that in silicon are still in flight through the hierarchy,
  producing results the RTL never yields.
- **Hierarchy is ownership, not a call graph.** `Core` owns its units and
  `Processor` owns `Memory` for construction and lifetime; it is never
  walked upward (`child->parent()->…`) or sideways into a sibling.
- **A side effect lives in the unit that owns it.** `Memory` counts its own
  DRAM reads and writes as it receives them; a strobe to another block is an
  event link (§4).

---

## 1. Units

A unit derives from `SimObject<U>` and reacts in one of two ways, never both:

| style | when | the unit defines |
|---|---|---|
| handler ports | one input per action, no state that changes with time | `bool handler(T& pkt)` per input port |
| `on_tick()` | several inputs, arbitration, or internal timed state | `void on_tick()` and, to sleep when idle, `bool busy()` |

The choice is checked at construction: a unit with `on_tick()` reads plain
ports (`SimPortIn<T>{this, capacity}`), a unit without it has only handler ports.
The hooks (`on_tick()`, `busy()`, `on_reset()`) may be protected when the unit declares
`friend class SimObject<U>;`, and an intermediate base `B : SimObject<B>` may
declare them virtual for its subclasses (`FuncUnit`).

```cpp
class Alu : public SimObject<Alu> {
public:
  SimPortIn<instr_trace_t*>  input{this, 2, &Alu::execute};
  SimPortOut<instr_trace_t*> output{this};
  using SimObject::SimObject;                       // (const SimContext&, name)

  bool execute(instr_trace_t*& trace) {
    if (output.full()) { return false; }            // leave it; retried when output frees
    output.send(trace, latency(trace));
    return true;                                    // taken: the port pops it
  }
};
auto alu = Alu::Create("alu");
alu->output.bind(&next->input);
```

A handler is called for the packet at the head of its port, at most once
per cycle. Returning `true` pops it; returning `false` leaves it, and the
port retries next cycle, or, when an output's `full()` refused during the
handler, once that output frees a slot. `SimPortInArray<T>{this, n, capacity,
&U::handler}` gives `n` handler ports served by `bool handler(uint32_t i,
T& pkt)`.

An `on_tick()` unit reads its ports itself:

```cpp
class Slow : public SimObject<Slow> {
public:
  SimPortIn<int>  input{this, 2};
  SimPortOut<int> output{this};
  using SimObject::SimObject;

  void on_tick() {
    uint64_t now = SimPlatform::instance().cycles();
    if (busy_ && now == done_at_)  { output.send(value_ * 10, 0); busy_ = false; }
    if (!busy_ && !input.empty())  { value_ = input.pop(); done_at_ = now + 5; busy_ = true; }
  }
  bool busy() { return busy_; }   // a countdown keeps the unit awake

private:
  bool busy_ = false;
  int value_ = 0;
  uint64_t done_at_ = 0;
};
```

Without `busy()`, an `on_tick()` unit is always busy: it runs every cycle,
as a clocked RTL block does. Defining `busy()` lets it sleep: `on_tick()`
then runs only on cycles where one of the unit's ports holds a visible
packet or `busy()` is true, and on the cycles it is woken (§3). Return `true`
from `busy()` exactly while state beyond the ports changes with time: a
countdown, a cycle it waits for, a sequence it is emitting. Packets still in
flight and a full output do not count: their arrival and the freed slot wake
the unit. A unit with no such state (an arbiter, a buffer) returns `false`:
it works only on its inputs. A `busy()` that is false too early changes
cycle counts (the model-parity and perf gates catch it); one that is true too
long, or none at all, only costs speed. An integral over cycles (a latency sum of outstanding requests) uses
`Occupancy` instead of adding on every tick, so it does not keep the
unit awake.

The rest of the unit API:

- `U::Create(name, args...)` builds the unit through the platform, which
  holds it (`SimContext` is the platform's key, so units are built nowhere
  else). Units refer to each other by raw pointer, never `shared_ptr`.
- `on_reset()`, optional, runs in creation order on `SimPlatform::reset()` and
  on `reset_launch()` (§7). It clears queues and counters; ports keep their
  bindings.
- `tick_wake()` runs the unit again from the earliest legal point. It is for
  state changed outside the unit's tick and ports: a host call (`start()`,
  `dcr_write()`), an event-link handler, a `pop_callback` credit, or the unit
  that owns it.
- `name()` is the name given to `Create`; the current cycle is
  `SimPlatform::instance().cycles()`.

---

## 2. Ports

```cpp
SimPortIn<Req>  in{this, 4};                                  // read by on_tick()
SimPortIn<Req>  in{this, 4, &U::handle};                      // handler port
SimPortOut<Req> out{this};
SimPortInArray<Req>  ins{this, n, 2};   SimPortOutArray<Req> outs{this, n};

in.set_registered_ready(true);   // before simulation: slot reusable next cycle only
out.bind(&other->in, 3);     // once, at elaboration; 3 = the latency of out.send(pkt)
```

| `SimPortOut<T>` (writer) | |
|---|---|
| `full()` | the reader has no room now; on `true` the running unit or handler is woken when a slot frees |
| `send(pkt)`, `send(pkt, latency)` | requires `!full()`; the bind latency, or an explicit one |
| `try_send(pkt[, latency])` | `full()` then `send`; returns whether it sent |
| `size()`, `capacity()` | packets sent and not yet taken by the reader, and the reader's room |

| `SimPortIn<T>` (reader) | |
|---|---|
| `empty()`, `peek()`, `pop()` | the oldest packet visible this cycle |
| `head_cycle()` | the cycle the oldest packet becomes visible |
| `size()`, `capacity()`, `full()` | packets in flight and visible, and the room |
| `push(pkt, latency)` | stage a packet into the owner's own port (requires `!full()`) |
| `bypass_to(out[, conv])` | make the port a wire to `out`'s reader |
| `set_registered_ready(bool)` | a slot popped this cycle is offered next cycle only (before simulation) |
| `pop_callback(fn)` | run `fn` on every pop, in the reader's context |
| `ready_mask(first, count)` | (`SimPortInArray`) bit *k* set when port `first + k` has a visible packet |

`SimPortInArray` and `SimPortOutArray` index with `[i]` or the bounds-checked
`at(i)`, have `size()` and iterate with range-for.

**Internal pipeline registers.** A unit's own queue needs no class: it is an
`SimPortIn<T>` the unit fills with `push(pkt, latency)` and reads when the packet
becomes visible, so a multi-cycle pipeline is a port, not a deque of
timestamps. Visibility, capacity and wake-up follow the same rules as a
packet sent by another unit.

**Wires.** `in.bypass_to(out)` turns `in` into a wire: producers bound to
`in` reach `out`'s reader directly, with their own bind latency and none
added, and the unit never reads `in`. `in.bypass_to(out, conv)` converts
each packet on the way (`conv(T) -> V`, for `SimPortOut<V>`): an adapter that
repacks or retags without ticking. Only a plain port (of an `on_tick()` unit)
becomes a wire; §6 shows where. A `SimPortOut` follows the wires to
its reader once, on first use, so a wire costs nothing per packet.

**Credits.** `in.pop_callback(fn)` is the hook for a credit return: `fn` runs on
every pop of `in`, in the reader's context, and typically gives the
producer its credit back and wakes it. `RegSlice` (§5) returns its credits
this way across a domain boundary.

---

## 3. Timing rules

- **Order.** Within a cycle, units and handler ports run in creation order:
  a handler port as it is constructed, a unit with `on_tick()` when `Create()`
  returns, after its ports and the units its constructor made.
- **Visibility.** A packet sent at cycle *c* with latency *d* is visible at
  *c + d*; packets visible on the same cycle keep their send order. Latency
  0 is visible at once to units after the sender in creation order, next
  cycle to the others. Use it only for a genuinely combinational path.
- **Capacity** counts the packets in flight toward a port *and* the ones
  visible in it, so a producer cannot oversubscribe a port by sending with a
  long latency. A slot popped this cycle can be refilled the same cycle by a
  unit after the reader, unless the port has a registered ready
  (`set_registered_ready(true)`), which offers it next cycle (an RTL ready
  driven from a register).
- **Throughput.** A handler port takes at most one packet per cycle; a
  `on_tick()` unit pops as many as its code does.

The kernel visits only awake units, in creation order. A unit is woken by:

1. **arrival**: a packet becomes visible on one of its ports (it is woken on
   the visible cycle, not at the send);
2. **a freed slot**: an output whose `full()` returned `true` while the
   unit ran frees a slot (under a registered ready, the cycle after the pop);
3. **`tick_wake()`**, from a host call, an event-link handler or a credit;
4. **reset**, which wakes every unit.

A wake by a unit ahead of it in creation order runs it the same cycle; one
by a unit behind it, the next cycle, exactly as if every unit were ticked
each cycle and tested a flag. An `on_tick()` unit without `busy()` stays awake;
one with `busy()` stays awake while a plain port holds a visible packet or
`busy()` is true; a handler port stays awake
while its port holds a visible packet and its handler is not blocked on an
output.

---

## 4. Event links

`SimEventLink<Msg>` is the control-plane counterpart of a port: a one-way,
typed strobe for sporadic events (doorbells, barrier arrive and resume,
completion kicks). The receiver binds a member-function handler to its end;
the sender's end is bound to the receiver's (fan-in allowed).

```cpp
SimEventLink<GbarArrive> gbar_arrive_in{this};                  // Cluster
gbar_arrive_in.bind(this, &Cluster::on_gbar_arrive);           // void on_gbar_arrive(const GbarArrive&)

SimEventLink<GbarArrive> gbar_arrive_out{this};                 // Core
core->gbar_arrive_out.bind(&cluster->gbar_arrive_in);          // elaboration
gbar_arrive_out.send({bar_id, count, core_id});                 // delay 1
```

`send(msg, delay)` cannot fail or be refused, and `delay >= 1`. The handler
runs in the receiver's execution domain at the start of cycle
`now + delay`, before that domain's units tick, in a canonical order (due
cycle, source domain with the host last, send order), so serial and
parallel runs see the same sequence. A link has no queue and no
back-pressure; a handler that changes state its unit's `on_tick()` reads calls
`tick_wake()`. Messages are self-contained values, never pointers to mutable
state.

| | `SimEventLink` | port |
|---|---|---|
| traffic | sporadic strobe | stream |
| refusable | no | yes: `full()` is `true` when full |
| consumer | the bound handler is called | handler port or `on_tick()` |
| occupancy | none | capacity and latency are the timing model |
| hardware analog | a strobe wire | a valid/ready pipe with a FIFO |

An event link and a port are the only ways to act on a unit in another
execution domain (§7).

---

## 5. Library units

[`simobject_lib.h`](../sim/common/simobject_lib.h) holds the reusable
pieces. Each moves at most one packet per output per cycle. The arbiters'
and crossbars' `delay` defaults to `default_delay(inputs, outputs)`: one
cycle (a register) when either side is wider than 2, zero otherwise.

| unit | behavior |
|---|---|
| `TxBuffer<T>` | registered elastic buffer: `In` holds `size` entries; the head moves to `Out` one cycle later |
| `DelayBuffer<T>` | fixed latency at full throughput: a packet reaching `In` at *t* arrives at *t + latency*; a full buffer back-pressures |
| `RegSlice<T>` | registered, credit-gated boundary stage (§7): `delay >= 1`, credits returned through `pop_callback` one cycle later |
| `TxArbiter<T>` | `Inputs` to `Outputs`: output *o* serves inputs `[o·R, o·R + R)`, *R* the next power of two of ⌈inputs / outputs⌉; each output packet is a `RspType{data, input}` |
| `TxCrossBar<T>` | each input goes to output `output_sel(pkt)`; priority or round robin per output; outputs are `RspType{data, input}`; `collisions()` counts contended grants |
| `TxRxArbiter<Req, Rsp>` | `ReqIn`/`RspOut` to `ReqOut`/`RspIn`: arbitrates requests, appending the group index to `tag`; responses route back on those bits, which are stripped |
| `TxRxCrossBar<Req, Rsp>` | a `TxCrossBar` on requests, appending the input index to `tag`; one shared arbiter picks each input's response |

An arbiter with as many inputs as outputs, and a 1×1 crossbar, make their
inputs wires (`bypass_to`) and add neither latency nor work. The helpers:
`Arbiter` is a value-type arbiter over up to 64 requesters, taking the
requests as a mask or a bit set (`select(requests)`, then `commit(i)` when
the grant is accepted), for the policies of `ArbiterType` (`Priority`,
`RoundRobin`, `Matrix`, `GreedyRR`, `GreedyMatrix`); `Occupancy` is the cycle
integral of §1 (`add(delta)`, `total()`).

A library unit is created like any unit and bound through its public ports;
its container keeps the `Ptr` (§6). The examples below come from the models.

**Buffers.** A fixed latency on `bind(&in, latency)` holds the packets in
flight in the receiver's port, so a 2-entry port sustains two packets per
latency. A `DelayBuffer` holds them itself and runs at full throughput; a
`TxBuffer` adds a registered elastic stage (`socket.cpp`, `core.cpp`):

```cpp
auto req_link = DelayBuffer<MemReq>::Create(sname, VX_CFG_L1_LINK_LATENCY);
auto rsp_link = DelayBuffer<MemRsp>::Create(sname, VX_CFG_L1_LINK_LATENCY);
req_src.bind(&req_link->In);   req_link->Out.bind(&req_dst);
rsp_src.bind(&rsp_link->In);   rsp_link->Out.bind(&rsp_dst);

auto buf = TxBuffer<MemReq>::Create(sname, 2);       // 2 entries
lsu_req_out.bind(&buf->In);
buf->Out.bind(&dcache_req_in);
```

**Boundary stage.** A `RegSlice` crosses execution domains (§7). It is
created in the sending domain; its output is the only port the receiving
domain sees (`cluster.cpp`):

```cpp
RegSlice<RasterReq>::Ptr req_slice;
{
  SimPlatform::DomainScope core_scope(core);           // the sender's domain
  req_slice = RegSlice<RasterReq>::Create(sname, 1);   // delay >= 1
}
sfu->raster_req_out.bind(&req_slice->In);
req_slice->Out.bind(&raster_bus->ReqIn[cid]);
```

**One-way arbiter.** `TxArbiter` and `TxCrossBar` emit `RspType{data, input}`,
so the receiving port has that type; `input` names the input that won:

```cpp
using TraceArbiter = TxArbiter<instr_trace_t*>;
auto arb = TraceArbiter::Create(sname, ArbiterType::RoundRobin, 4, 1);   // delay: default_delay(4, 1) = 1
for (uint32_t i = 0; i < 4; ++i) {
  slots[i]->output.bind(&arb->Inputs[i]);
}
arb->Outputs[0].bind(&unit->input);    // SimPortIn<TraceArbiter::RspType> input{this, 2};

// in the receiving unit:
auto rsp = input.pop();
instr_trace_t* trace = rsp.data;       // rsp.input: the slot it came from
```

**One-way crossbar.** `output_sel` maps a packet to its output; the outputs
are a power of two. `collisions()` counts the grants another input contended
for:

```cpp
auto xbar = TxCrossBar<MemReq>::Create(sname, ArbiterType::RoundRobin, 4, 8,
    [](const MemReq& req) { return uint32_t((req.addr >> 6) & 7); });   // bank by line address
```

**Request/response arbiter.** `TxRxArbiter` arbitrates `R` inputs per
output and routes each response back by its tag. A request from group slot
`r` leaves with `tag = (tag << log2 R) | r`; the unit downstream must copy
the request's `tag` into its response, and the arbiter strips those bits on
the way back, so the input sees its own tag again. The tag field needs
`log2 R` spare high bits per level (`cache_cluster.cpp`):

```cpp
using MemArbiter = TxRxArbiter<MemReq, MemRsp>;
auto mem_arb = MemArbiter::Create(sname, ArbiterType::RoundRobin, num_units, 1);
for (uint32_t i = 0; i < num_units; ++i) {
  caches[i]->mem_req_out.bind(&mem_arb->ReqIn[i]);
  mem_arb->RspOut[i].bind(&caches[i]->mem_rsp_in);
}
mem_arb->ReqOut[0].bind(&memory->req_in);
memory->rsp_out.bind(&mem_arb->RspIn[0]);
```

**Request/response crossbar.** `TxRxCrossBar` tags each request with its
input index (`log2 inputs` bits) and returns each response to that input
(`memory.cpp`):

```cpp
using MemCrossBar = TxRxCrossBar<MemReq, MemRsp>;
auto xbar = MemCrossBar::Create(sname, ArbiterType::RoundRobin, num_ports, num_banks,
    [lg2_block = log2ceil(block_size), num_banks](const MemReq& req) {
      return uint32_t((req.addr >> lg2_block) & (num_banks - 1));
    });
for (uint32_t i = 0; i < num_ports; ++i) {
  port_req_out[i].bind(&xbar->ReqIn[i]);
  xbar->RspOut[i].bind(&port_rsp_in[i]);
}
for (uint32_t b = 0; b < num_banks; ++b) {
  xbar->ReqOut[b].bind(&bank_req_in[b]);
  bank_rsp_out[b].bind(&xbar->RspIn[b]);
}
```

**`Arbiter` in a unit's own `on_tick()`.** Select, send, then commit, so a
refused grant leaves the order unchanged (`operands.cpp`):

```cpp
Arbiter merge_arb_{ArbiterType::RoundRobin, VX_CFG_NUM_OPCS};   // reset() in on_reset()

uint64_t requests = merge_in_.ready_mask();
if (requests != 0 && !Output.full()) {
  uint32_t g = merge_arb_.select(requests);
  Output.send(merge_in_[g].pop(), 1);
  merge_arb_.commit(g);
}
```

**`Occupancy`.** A latency sum of outstanding requests, without ticking
(`memory.cpp`):

```cpp
Occupancy pending_reads_;
pending_reads_.add(+1);                        // a read is accepted
pending_reads_.add(-1);                        // its response leaves
perf.latency = pending_reads_.total();         // sum over cycles of the reads outstanding
```

---

## 6. Containers

A container (`Core`, `Socket`, `Cluster`, `CacheCluster`, `Processor`)
builds its children and binds them; it does not forward packets. Its ports
are its children's own ports, exposed by pointer, so a crossing of the
hierarchy adds no unit, no tick and no latency:

```cpp
class CacheCluster : public SimObject<CacheCluster> {
public:
  std::vector<std::vector<SimPortIn<MemReq>*>> core_req_in;   // the input arbiters' ports
  std::vector<SimPortOut<MemRsp>*>            core_rsp_out;
  std::vector<SimPortOut<MemReq>*>            mem_req_out;    // the memory arbiters' ports
  std::vector<SimPortIn<MemRsp>*>             mem_rsp_in;
  ...
};

// CacheCluster constructor
core_req_in[j][i] = &input_arbs[i]->ReqIn[j];
mem_req_out[i]    = &mem_arbs[i]->ReqOut[0];

// the parent binds through them
l1->mem_req_out[i]->bind(l2->core_req_in[k][i]);
```

A container with no behavior of its own defines no `on_tick()` and costs
nothing per cycle. An `on_tick()` unit whose work reduces to a wire in some
configurations keeps its ports and turns them into wires with `bypass_to`:
an arbiter with one input per output, or `LsuMemAdapter` with one lane,
whose request and response ports become converting wires
(`LsuReq` → `MemReq`, `MemRsp` → `LsuRsp`).

---

## 7. Execution domains and multi-threading

The platform partitions the design into **execution domains**. Topology
containers (Socket, Cluster) open a `SimPlatform::DomainScope` per
partition, and every unit created inside inherits it. Domain 0 is the
uncore (memory system, KMU, host-facing blocks); each socket gets its own.
Leaf units never manage domains.

Defining `SIMX_MT=<T>` in the build's `CONFIGS` runs the domains on `T`
worker threads (default serial). `sim/simx/types.h` resolves the macro into
`SIMX_NUM_WORKERS`, which the application passes to
`SimPlatform::set_num_workers()` before the first tick. **Cycles are
bit-identical for every thread count:** a domain's ports, wake-ups and scan
are touched only by its own thread, cross-domain deliveries run in canonical
order, and cycles are fenced by barriers. A model that follows the cardinal
rule gets deterministic parallelism for free.

**Crossing a domain.** A send into another domain needs a latency of at
least one cycle, and every output whose reader is in another domain must be
a `RegSlice` output (`SimPortOut::mark_boundary()`), owned by the sending domain:
its stall decision reads only its own credit count, never the remote port.
The topology is validated at the first reset; an unregistered crossing is
reported (`SIMX-MT: unregistered cross-domain edge`) and caps execution at
one thread. Cross-domain packets and calls travel through per-(target,
source) mailbox lanes, triple-buffered by epoch, so no lock or shared
counter sits on the path.

**Host loops.** `SimPlatform::run_until(stop)` equals
`do { tick(); } while (!stop());` cycle for cycle; with workers, the stop
check runs in the release step of the cycle-end barrier, so a cycle costs
one barrier instead of `tick()`'s two. `run_until(stop, voters, busy)` is for
a loop that cannot end while any voter is busy: after each cycle the thread
ticking a voter evaluates `busy(voter)`, and `stop(busy)` gets their OR;
with `busy` set it must return `false` and touch only uncore state. With
workers, this replaces the barrier with **neighbor synchronization**:

- A domain publishes the cycles it has completed and its vote, and starts
  cycle *c* once every neighbor has completed *c*. Neighbors are the domains
  a port joins, plus the uncore with every domain, so no socket gets more
  than two cycles ahead of the slowest.
- Thread 0, which owns the uncore, decides the stop check for cycle *c* when
  it starts the next. A thread with a busy domain may run ahead; a thread
  whose domains all voted idle waits for the decision. When every vote is
  idle every other thread is parked, so `stop(false)` sees exactly the state
  `run_until(stop)` would, and the stop cycle and every counter match.
- A cross-domain call (an event link included) between two domains no port
  joins aborts the run: the synchronization could not order it.

`Processor::run()` lets its cores vote "running" (`Core::running()`) when
the processor has one cluster and no VM; other host loops (cache flushes,
copy-engine transfers) keep the barrier. In the functional kernel the unit
of synchronization is a quantum instead of a cycle (§8).

**Quiescence.** `SimPlatform::idle()` is true when no packet is in flight
or held in a port and no cross-domain call is pending. Host flush and drain
loops use it rather than testing units' internals; resets happen only from
quiescence.

**Kernel launch.** `SimPlatform::reset_launch()` resets every unit except
those built under a `SimPlatform::LaunchKeepScope` (the last-level cache and
the memory below it when the command processor writes into that cache):
the hardware does not reset between a transfer and the launch that follows,
so the transfer's fills may still be in flight there. While it runs,
`launch_resetting()` is true, so a unit resets its launch state and keeps
what only a device reset clears.

What makes a unit MT-safe is the absence of illegal code. The kernel's
asserts and topology check catch illegal crossings, and
[`ci/check_simx_mt_boundary.sh`](../ci/check_simx_mt_boundary.sh) rejects
threading vocabulary in `sim/simx/`:

| a unit may | a unit may not |
|---|---|
| change its own state | touch another unit's fields or call its methods across a domain |
| send on its own outputs (across domains: through a `RegSlice`, latency ≥ 1) | read `full()`/`size()` of a port in another domain |
| declare event-link ends, bind handlers, send on its own ends | use `cross_call`, `run_in_domain`, `domain()`, `std::atomic`, threads or locks |
| (host context, between cycles) read any unit's state | hand-roll quiescence tests: use `idle()` |

Host code between cycles changes units only through their API (`reset`,
`start`, `dcr_write`), the same rule testbenches follow.

---

## 8. Functional kernel

Building with `CONFIGS="-DSIMX_FUNCTIONAL"`, in a build tree of its own such
as `build32_functional/`, keeps the executor and drops the timing, for
full-speed architectural runs (conformance suites, ISA bring-up). Model code
is the same; the kernel's symbols live in their own inline namespace, so a
timed and a functional object file fail to link together.

- Every latency above one cycle becomes one; latency 0 keeps its meaning.
- Ports never fill: `full()` is always `false`, `capacity()` is nominal
  (2³⁰), and a port's storage grows on demand up to a high-water assert.
- Domains, canonical ordering, topology validation and `SIMX_MT` are kept;
  results are bit-identical for every thread count.
- `-DSIMX_QUANTUM=<Q>` (passed to `SimPlatform::set_quantum()`) lets domains
  run `Q` cycles between synchronizations; a cross-domain call runs at the
  first quantum boundary at or after its due cycle. `Q` changes the
  (non-physical) cycle count but not its determinism; around 100 is a good
  start. The timed kernel rejects `Q > 1`.
- Cycle counts are monotonic but **non-physical**: a functional build never
  feeds `perf_gate`, `model_parity` or baseline regeneration.
