// Copyright © 2026  Vortex GPGPU
// SPDX-License-Identifier: MIT
//
// Conformance of the simobject.h port timing.
//
// Each scenario is a small graph of nodes driven by scripts written against
// an abstract port interface, and runs as:
//   all     every unit asks to run again every cycle;
//   idle    units define busy() and ask through it for exactly
//           the activity a real model would (while producing, and until a
//           cycle they wait for);
//   handler (where a scenario has forwarding nodes) those nodes are native
//           handler-port units with the latency on the link.
// Every send, refused send and pop is logged with its cycle. The `all` log
// must hash to the golden value, recorded from the original channel kernel
// that ticked every object each cycle; `idle` and `handler` must match it on
// sends and pops (a blocked writer is not ticked, so its refused retries are
// not there to log).

#include <cstdio>
#include <functional>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

#include "simobject.h"

namespace {

using Log = std::vector<std::string>;

// What a script sees: its node's ports, the clock and the wake-up requests.
struct Ctx {
  virtual ~Ctx() = default;
  virtual uint64_t now() const = 0;
  virtual bool in_empty(uint32_t port) = 0;
  virtual int in_pop(uint32_t port) = 0;
  virtual bool out_ready(uint32_t port) = 0;
  virtual void out_send(uint32_t port, int value, uint32_t delay) = 0;
  // Activity requests: ignored by `all`, honored by `idle`.
  virtual void again() = 0;
  virtual void wake_at(uint64_t cycle) = 0;

  uint32_t node = 0;
  Log* log = nullptr;

  void put(const char* what, int value, uint32_t port, uint32_t delay = 0) {
    std::ostringstream os;
    os << "c" << this->now() << " n" << node << " " << what << " p" << port << " v" << value;
    if (delay != ~0u) {
      os << " d" << delay;
    }
    log->push_back(os.str());
  }

  // Logged helpers the scripts use.
  bool try_send(uint32_t port, int value, uint32_t delay) {
    if (!this->out_ready(port)) {
      this->put("full", value, port, ~0u);
      return false;
    }
    this->out_send(port, value, delay);
    this->put("send", value, port, delay);
    return true;
  }

  bool try_pop(uint32_t port, int* value = nullptr) {
    if (this->in_empty(port)) {
      return false;
    }
    int v = this->in_pop(port);
    this->put("pop", v, port, ~0u);
    if (value != nullptr) {
      *value = v;
    }
    return true;
  }
};

using Script = std::function<void(Ctx&)>;

// An edge: out port `src_port` of node `src` feeds in port `dst_port` of `dst`.
struct Edge {
  uint32_t src, src_port, dst, dst_port;
};

struct InSpec {
  uint32_t capacity = 2;
  bool registered_ready = false;
};

struct Scenario {
  const char* name;
  uint64_t cycles;
  std::vector<Script> nodes;            // index = registration (tick) order
  std::vector<std::vector<InSpec>> ins; // per node, its input ports
  std::vector<Edge> edges;
};

constexpr uint32_t kMaxPorts = 2;

// ---------------------------------------------------------------- runner

using namespace vortex;

uint64_t g_activations = 0;

class V3Node : public SimObject<V3Node> {
public:
  std::vector<std::unique_ptr<SimPortIn<int>>> in;
  std::vector<std::unique_ptr<SimPortOut<int>>> out;
  Script script;
  bool idle_mode = false;
  bool want_again = false;
  struct C;
  std::unique_ptr<Ctx> ctx;

  V3Node(const vortex::SimContext& c, const std::string& name, const std::vector<InSpec>& specs)
    : SimObject(c, name) {
    for (auto& s : specs) {
      in.emplace_back(new SimPortIn<int>(this, s.capacity));
      in.back()->set_registered_ready(s.registered_ready);
    }
    for (uint32_t p = 0; p < kMaxPorts; ++p) {
      out.emplace_back(new SimPortOut<int>(this));
    }
  }

  void on_tick() {
    ++g_activations;
    want_again = false;
    wait_until = 0;
    script(*ctx);
  }

  bool busy() { return !idle_mode || want_again || wait_until > SimPlatform::instance().cycles(); }

  uint64_t wait_until = 0;
};

struct V3Node::C : Ctx {
  V3Node* n;
  explicit C(V3Node* node) : n(node) {}
  uint64_t now() const override { return SimPlatform::instance().cycles(); }
  bool in_empty(uint32_t p) override { return n->in[p]->empty(); }
  int in_pop(uint32_t p) override { return n->in[p]->pop(); }
  bool out_ready(uint32_t p) override { return !n->out[p]->full(); }
  void out_send(uint32_t p, int v, uint32_t d) override { n->out[p]->send(v, d); }
  void again() override { n->want_again = true; }
  void wake_at(uint64_t cycle) override {
    if (n->wait_until == 0 || cycle < n->wait_until) {
      n->wait_until = cycle;
    }
  }
};

// A forwarder as a handler port, the latency on the link.
class Forwarder : public SimObject<Forwarder> {
public:
  std::unique_ptr<SimPortIn<int>> input;
  SimPortOut<int> output{this};
  uint32_t node = 0;
  uint32_t latency = 1;
  Log* log = nullptr;

  Forwarder(const vortex::SimContext& c, const std::string& name, const InSpec& spec)
    : SimObject(c, name)
    , input(new SimPortIn<int>(this, spec.capacity, &Forwarder::forward)) {
    input->set_registered_ready(spec.registered_ready);
  }

  bool forward(int& v) {
    ++g_activations;
    if (output.full()) {
      return false;
    }
    this->put("pop", v, ~0u);
    output.send(v);
    this->put("send", v, latency);
    return true;
  }

private:
  void put(const char* what, int v, uint32_t delay) {
    std::ostringstream os;
    os << "c" << SimPlatform::instance().cycles() << " n" << node << " " << what << " p0 v" << v;
    if (delay != ~0u) {
      os << " d" << delay;
    }
    log->push_back(os.str());
  }
};

// native[i] >= 0: node i is a Forwarder with that link latency.
Log run_v3(const Scenario& sc, bool idle_mode, uint64_t* ticks, const std::vector<int>& native = {}) {
  Log log;
  auto& platform = vortex::SimPlatform::instance();
  g_activations = 0;
  std::vector<std::shared_ptr<V3Node>> nodes(sc.nodes.size());
  std::vector<std::shared_ptr<Forwarder>> fwds(sc.nodes.size());
  for (uint32_t i = 0; i < sc.nodes.size(); ++i) {
    std::string name = "n" + std::to_string(i);
    if (i < native.size() && native[i] >= 0) {
      fwds[i] = Forwarder::Create(name, sc.ins[i].at(0));
      fwds[i]->node = i;
      fwds[i]->log = &log;
      fwds[i]->latency = uint32_t(native[i]);
      continue;
    }
    nodes[i] = V3Node::Create(name, sc.ins[i]);
    auto* c = new V3Node::C(nodes[i].get());
    c->node = i;
    c->log = &log;
    nodes[i]->ctx.reset(c);
    nodes[i]->script = sc.nodes[i];
    nodes[i]->idle_mode = idle_mode;
  }
  auto in_of = [&](uint32_t n, uint32_t p) { return fwds[n] ? fwds[n]->input.get() : nodes[n]->in[p].get(); };
  for (auto& e : sc.edges) {
    if (fwds[e.src]) {
      fwds[e.src]->output.bind(in_of(e.dst, e.dst_port), fwds[e.src]->latency);
    } else {
      nodes[e.src]->out[e.src_port]->bind(in_of(e.dst, e.dst_port));
    }
  }
  platform.reset();
  for (uint64_t c = 0; c < sc.cycles; ++c) {
    platform.tick();
  }
  *ticks = g_activations;
  nodes.clear();
  fwds.clear();
  platform.finalize();
  return log;
}

// ---------------------------------------------------------------- scenarios

// A reader that pops everything ready on port 0, one per tick.
Script reader() {
  return [](Ctx& c) { c.try_pop(0); };
}

// A reader that pops at most one packet on cycles that are multiples of `n`.
Script slow_reader(uint64_t n) {
  return [n](Ctx& c) {
    if (c.now() % n == 0) {
      c.try_pop(0);
    }
    c.wake_at((c.now() / n + 1) * n);
  };
}

// Sends `count` values, one per cycle from `start`, retrying while full.
Script stream(int count, uint64_t start, uint32_t delay) {
  auto sent = std::make_shared<int>(0);
  return [=](Ctx& c) {
    if (c.now() < start) {
      c.wake_at(start);
      return;
    }
    if (*sent < count && c.try_send(0, 100 + *sent, delay)) {
      ++*sent;
    }
    if (*sent < count && c.out_ready(0)) {
      c.again();
    }
  };
}

// Sends a list of {cycle, value, delay} on port `port`.
struct Shot {
  uint64_t cycle;
  int value;
  uint32_t delay;
};
Script shots(std::vector<Shot> list, uint32_t port = 0) {
  return [list, port](Ctx& c) {
    uint64_t next = ~uint64_t(0);
    for (auto& s : list) {
      if (s.cycle == c.now()) {
        c.try_send(port, s.value, s.delay);
      } else if (s.cycle > c.now() && s.cycle < next) {
        next = s.cycle;
      }
    }
    if (next != ~uint64_t(0)) {
      c.wake_at(next);
    }
  };
}

// Pops from port 0 and forwards with `delay` on port 0.
Script forward(uint32_t delay) {
  return [delay](Ctx& c) {
    if (c.in_empty(0) || !c.out_ready(0)) {
      return;
    }
    int v = 0;
    c.try_pop(0, &v);
    c.out_send(0, v, delay);
    c.put("send", v, 0, delay);
  };
}

std::vector<Scenario> scenarios() {
  std::vector<Scenario> s;
  auto one = std::vector<InSpec>{InSpec{}};
  auto none = std::vector<InSpec>{};

  s.push_back({"delay0, reader after sender", 6,
               {shots({{1, 1, 0}, {3, 2, 0}}), reader()}, {none, one}, {{0, 0, 1, 0}}});
  s.push_back({"delay0, reader before sender", 6,
               {reader(), shots({{1, 1, 0}, {3, 2, 0}})}, {one, none}, {{1, 0, 0, 0}}});
  s.push_back({"delay 3 then delay 1, same arrival", 8,
               {shots({{0, 1, 3}, {2, 2, 1}}), reader()}, {none, one}, {{0, 0, 1, 0}}});
  s.push_back({"delay 1 then delay 3 then delay 2", 10,
               {reader(), shots({{0, 1, 4}, {1, 2, 1}, {2, 3, 2}})},
               {std::vector<InSpec>{InSpec{4, false}}, none}, {{1, 0, 0, 0}}});
  s.push_back({"capacity counts in-flight packets", 10,
               {shots({{0, 1, 5}, {0, 2, 5}, {0, 3, 5}, {1, 4, 1}, {5, 5, 1}, {6, 6, 1}}), reader()},
               {none, one}, {{0, 0, 1, 0}}});
  for (bool rr : {false, true}) {
    auto in = std::vector<InSpec>{InSpec{2, rr}};
    s.push_back({rr ? "stream, writer first, registered ready" : "stream, writer first", 24,
                 {stream(8, 0, 1), slow_reader(3)}, {none, in}, {{0, 0, 1, 0}}});
    s.push_back({rr ? "stream, reader first, registered ready" : "stream, reader first", 24,
                 {slow_reader(3), stream(8, 0, 1)}, {in, none}, {{1, 0, 0, 0}}});
    s.push_back({rr ? "stream delay 0, reader first, registered ready" : "stream delay 0, reader first", 24,
                 {slow_reader(2), stream(8, 0, 0)}, {in, none}, {{1, 0, 0, 0}}});
  }
  s.push_back({"delay-0 chain forward", 8,
               {shots({{0, 1, 0}, {1, 2, 0}, {2, 3, 0}}), forward(0), reader()},
               {none, one, one}, {{0, 0, 1, 0}, {1, 0, 2, 0}}});
  s.push_back({"delay-0 chain backward", 8,
               {reader(), forward(0), shots({{0, 1, 0}, {1, 2, 0}, {2, 3, 0}})},
               {one, one, none}, {{2, 0, 1, 0}, {1, 0, 0, 0}}});
  s.push_back({"fan-in, two writers, mixed delays", 10,
               {shots({{0, 1, 1}, {1, 2, 0}, {2, 3, 2}}), reader(), shots({{0, 11, 1}, {1, 12, 1}, {3, 13, 0}})},
               {none, std::vector<InSpec>{InSpec{4, false}}, none}, {{0, 0, 1, 0}, {2, 0, 1, 0}}});
  s.push_back({"pipeline with latency 3 and backpressure", 30,
               {stream(10, 1, 1), forward(3), slow_reader(2)},
               {none, one, one}, {{0, 0, 1, 0}, {1, 0, 2, 0}}});
  return s;
}

// Scenarios whose forward() nodes have a native handler-port equivalent.
std::vector<int> native_forwarders(const std::string& name) {
  if (name == "delay-0 chain forward") {
    return {-1, 0, -1};
  }
  if (name == "delay-0 chain backward") {
    return {-1, 0, -1};
  }
  if (name == "pipeline with latency 3 and backpressure") {
    return {-1, 3, -1};
  }
  return {};
}

Log observable(const Log& log) {
  Log out;
  for (auto& line : log) {
    if (line.find(" full ") == std::string::npos) {
      out.push_back(line);
    }
  }
  return out;
}

struct Gold {
  const char* name;
  uint64_t timed;
  uint64_t functional;
};

const Gold kGold[] = {
  {"delay0, reader after sender", 0x1661522c6244df45ull, 0x1661522c6244df45ull},
  {"delay0, reader before sender", 0xa9667e1132100d91ull, 0xa9667e1132100d91ull},
  {"delay 3 then delay 1, same arrival", 0x3a06428fb3e016c4ull, 0x9f7ba0a84d9a851full},
  {"delay 1 then delay 3 then delay 2", 0xcf9e7f3155722396ull, 0xde4550e5738f679full},
  {"capacity counts in-flight packets", 0xc0e9bc2b0b0fdd08ull, 0xddad2a9be9bbb3d4ull},
  {"stream, writer first", 0xfd92d55586fef315ull, 0xfb8c71acae0b58c9ull},
  {"stream, reader first", 0x277242e9f3b1f53eull, 0xe343f13f0a3070e8ull},
  {"stream delay 0, reader first", 0x98b35aba31283675ull, 0x1c70011704a666e7ull},
  {"stream, writer first, registered ready", 0xfd92d55586fef315ull, 0xfb8c71acae0b58c9ull},
  {"stream, reader first, registered ready", 0x90708936ece4e82eull, 0xe343f13f0a3070e8ull},
  {"stream delay 0, reader first, registered ready", 0xc0c528103c8be20dull, 0x1c70011704a666e7ull},
  {"delay-0 chain forward", 0x94d003c265eca659ull, 0x94d003c265eca659ull},
  {"delay-0 chain backward", 0xda485e56744f3fefull, 0xda485e56744f3fefull},
  {"fan-in, two writers, mixed delays", 0xbf5c64989be35b68ull, 0xd0a8a9ac69a56354ull},
  {"pipeline with latency 3 and backpressure", 0xb1bd0a6d8fa54151ull, 0x57067e513d22c304ull},
};

uint64_t hash_of(const Log& log) {
  uint64_t h = 1469598103934665603ull;
  for (auto& line : log) {
    for (char ch : line) {
      h = (h ^ uint8_t(ch)) * 1099511628211ull;
    }
    h = (h ^ 10) * 1099511628211ull;
  }
  return h;
}

}  // namespace

int main() {
  int failures = 0;
  // Scripts keep state, so each run gets freshly built scenarios.
  size_t count = scenarios().size();
  for (size_t i = 0; i < count; ++i) {
    Scenario sc = scenarios()[i];
    uint64_t ticks_all = 0;
    uint64_t ticks_idle = 0;
    Log b = run_v3(sc, false, &ticks_all);
    Log c = run_v3(scenarios()[i], true, &ticks_idle);
    uint64_t gold = vortex::kSimFunctional ? kGold[i].functional : kGold[i].timed;
    bool ok = (hash_of(b) == gold) && (observable(b) == observable(c));
    // Forwarder nodes rewritten as native handler units must not change
    // what happens.
    auto native = native_forwarders(sc.name);
    uint64_t ticks_native = 0;
    if (!native.empty()) {
      Log d = run_v3(scenarios()[i], true, &ticks_native, native);
      ok &= (observable(b) == observable(d));
    }
    std::printf("%-4s %-48s events=%-3zu ticks all=%-4lu idle=%-4lu%s\n", ok ? "PASS" : "FAIL",
                sc.name, b.size(), (unsigned long)ticks_all, (unsigned long)ticks_idle,
                native.empty() ? "" : (" handler=" + std::to_string(ticks_native)).c_str());
    if (!ok) {
      ++failures;
      size_t n = std::max(b.size(), c.size());
      std::printf("     %-28s %-28s\n", "all", "idle");
      for (size_t k = 0; k < n; ++k) {
        std::printf("     %-28s %-28s\n", k < b.size() ? b[k].c_str() : "", k < c.size() ? c[k].c_str() : "");
      }
    }
  }
  std::printf("%s: %d failing\n", failures ? "FAILED" : "PASSED", failures);
  return failures ? 1 : 0;
}
