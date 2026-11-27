// Copyright © 2026  Vortex GPGPU
// SPDX-License-Identifier: MIT
//
// The simobject.h API pieces the conformance suite does not reach: a unit
// with internal work (busy), a multi-cycle unit, a library arbiter,
// occupancy counters and hooks that are protected or virtual in a base.
// Built and run in both timed and functional (-DSIMX_FUNCTIONAL) builds.

#include <cstdio>
#include <string>
#include <vector>

#include "simobject_lib.h"

using namespace vortex;

namespace {

int g_failures = 0;

vortex::SimPlatform& platform() { return vortex::SimPlatform::instance(); }

void run(int cycles) {
  platform().reset();
  for (int i = 0; i < cycles; ++i) {
    platform().tick();
  }
}

void done() { platform().finalize(); }

void check(bool ok, const std::string& what) {
  std::printf("%s %s\n", ok ? "PASS" : "FAIL", what.c_str());
  if (!ok) {
    ++g_failures;
  }
}

// Emits `count` numbered packets, one per cycle.
class Source : public SimObject<Source> {
public:
  SimPortOut<int> output{this};
  int count = 0;
  int sent = 0;
  using SimObject::SimObject;

  void on_tick() {
    if (sent < count && !output.full()) {
      output.send(sent++);
    }
  }

  bool busy() { return sent < count && !output.full(); }  // a full output wakes us on pop
};

// Records what arrives and when.
class Sink : public SimObject<Sink> {
public:
  SimPortIn<int> input{this, 4, &Sink::take};
  std::vector<std::pair<uint64_t, int>> got;
  using SimObject::SimObject;

  bool take(int& v) {
    got.push_back({SimPlatform::instance().cycles(), v});
    return true;
  }
};

// Records what a TxArbiter delivers.
class ArbSink : public SimObject<ArbSink> {
public:
  SimPortIn<TxArbiter<int>::RspType> input{this, 4, &ArbSink::take};
  std::vector<std::pair<uint64_t, int>> got;
  using SimObject::SimObject;

  bool take(TxArbiter<int>::RspType& rsp) {
    got.push_back({SimPlatform::instance().cycles(), rsp.data});
    return true;
  }
};

// A multi-cycle operation: accept, finish 5 cycles later, then accept the next.
class Slow : public SimObject<Slow> {
public:
  SimPortIn<int> input{this, 2};
  SimPortOut<int> output{this};
  using SimObject::SimObject;

  void on_tick() {
    if (busy_ && SimPlatform::instance().cycles() == done_at_) {
      output.send(value_ * 10, 0);
      busy_ = false;
    }
    if (!busy_ && !input.empty()) {
      value_ = input.pop();
      done_at_ = SimPlatform::instance().cycles() + 5;
      busy_ = true;
    }
  }

  bool busy() { return busy_; }

private:
  bool busy_ = false;
  int value_ = 0;
  uint64_t done_at_ = 0;
};

// Counts outstanding requests: two from cycle 3, one leaves at 5; the total
// through cycle 9 is read on cycle 9.
class Counter : public SimObject<Counter> {
public:
  Occupancy outstanding;
  uint64_t total_at_9 = 0;
  using SimObject::SimObject;

  void on_tick() {
    if (SimPlatform::instance().cycles() == 3) {
      outstanding.add(+2);
    }
    if (SimPlatform::instance().cycles() == 5) {
      outstanding.add(-1);
    }
    if (SimPlatform::instance().cycles() == 9) {
      total_at_9 = outstanding.total();
    }
  }

  bool busy() { return SimPlatform::instance().cycles() < 9; }
};

// A base whose hooks are virtual and protected, as FuncUnit's: subclasses
// override them and are created with create_object<Subclass>.
class Stage : public SimObject<Stage> {
public:
  SimPortIn<int> input{this, 2};
  int resets = 0;
  std::vector<std::pair<uint64_t, int>> got;

  Stage(const SimContext& ctx, const std::string& name) : SimObject<Stage>(ctx, name) {}
  virtual ~Stage() = default;

protected:
  virtual void on_tick() = 0;
  virtual void on_reset() { ++resets; }

  friend class SimObject<Stage>;
};

class Doubler : public Stage {
public:
  using Stage::Stage;

protected:
  void on_tick() override {
    if (!input.empty()) {
      got.push_back({SimPlatform::instance().cycles(), input.pop() * 2});
    }
  }
};

}  // namespace

int main() {
  const char* mode = vortex::kSimFunctional ? "functional" : "timed";
  std::printf("== %s build\n", mode);

  // 1. Link latency: a 4-cycle path, no delay unit.
  {
    auto src_p = Source::Create("src");
    auto dst_p = Sink::Create("dst");
    auto& src = *src_p;
    auto& dst = *dst_p;
    src.count = 3;
    src.output.bind(&dst.input, 4);
    run(20);
    bool ok = dst.got.size() == 3;
    uint64_t first = ok ? dst.got[0].first : 0;
    uint64_t want = vortex::kSimFunctional ? 1 : 4;
    check(ok && first == want, "link latency: first arrival at cycle " + std::to_string(first) +
          " (want " + std::to_string(want) + ")");
    check(platform().idle(), "kernel idle once the work is done");
    done();
  }

  // 2. A multi-cycle unit: busy for 5 cycles per packet.
  {
    auto src_p = Source::Create("src");
    auto slow_p = Slow::Create("slow");
    auto dst_p = Sink::Create("dst");
    auto& src = *src_p;
    auto& dst = *dst_p;
    src.count = 3;
    src.output.bind(&slow_p->input, 1);
    slow_p->output.bind(&dst.input);
    run(40);
    std::string trace;
    for (auto& g : dst.got) {
      trace += "c" + std::to_string(g.first) + ":" + std::to_string(g.second) + " ";
    }
    bool values = dst.got.size() == 3 && dst.got[0].second == 0 && dst.got[1].second == 10 &&
                  dst.got[2].second == 20;
    check(values, "multi-cycle unit forwards all packets: " + trace);
    if (!vortex::kSimFunctional) {
      // Accepted at 1, done at 6, when it takes the next.
      bool timing = values && dst.got[0].first == 6 && dst.got[1].first == 11 && dst.got[2].first == 16;
      check(timing, "multi-cycle unit timing (want c6 c11 c16)");
    }
    check(platform().idle(), "kernel idle once the work is done");
    done();
  }

  // 3. A library arbiter: two sources onto one sink, round robin.
  {
    auto a = Source::Create("a");
    auto b = Source::Create("b");
    auto merge = TxArbiter<int>::Create("merge", ArbiterType::RoundRobin, 2, 1, 0);
    auto dst = ArbSink::Create("dst");
    a->count = 3;
    b->count = 3;
    a->output.bind(&merge->Inputs[0]);
    b->output.bind(&merge->Inputs[1]);
    merge->Outputs[0].bind(&dst->input);
    run(30);
    check(dst->got.size() == 6, "arbiter delivers all " + std::to_string(dst->got.size()) + "/6");
    done();
  }

  // 4. Occupancy.
  {
    auto counter = Counter::Create("counter");
    run(12);
    // cycles 3,4 hold 2, cycles 5..9 hold 1: 4 + 5 = 9
    check(counter->total_at_9 == 9, "occupancy integral: " + std::to_string(counter->total_at_9));
    done();
  }

  // 5. Hooks protected and virtual in an intermediate base.
  {
    auto src = Source::Create("src");
    auto stage = platform().create_object<Doubler>("doubler");
    src->count = 3;
    src->output.bind(&stage->input);
    run(10);
    bool values = stage->got.size() == 3 && stage->got[0].second == 0 && stage->got[2].second == 4;
    check(Stage::kTicks && Stage::kResets && values && stage->resets == 1,
          "virtual protected hooks: " + std::to_string(stage->got.size()) + " packets, " +
          std::to_string(stage->resets) + " reset");
    done();
  }

  std::printf("%s: %d failing\n", g_failures ? "FAILED" : "PASSED", g_failures);
  return g_failures ? 1 : 0;
}
