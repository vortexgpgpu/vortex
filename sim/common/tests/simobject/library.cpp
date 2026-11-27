// Copyright © 2026  Vortex GPGPU
// SPDX-License-Identifier: MIT
//
// simobject_lib.h: each library unit is driven by seeded random traffic,
// and every delivery to a reader and every response to a requester is logged
// with its cycle. The log's hash must match the one recorded from the types.h
// units these replaced, under the same traffic (timed kernel).

#include <cstdio>
#include <deque>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

#include "simobject_lib.h"

namespace {

using namespace vortex;
using Log = std::vector<std::string>;

struct Pkt {
  uint32_t tag = 0;
  uint32_t val = 0;
  uint32_t addr = 0;
};

[[maybe_unused]] std::ostream& operator<<(std::ostream& os, const Pkt& p) {
  return os << "tag=" << p.tag << " val=" << p.val;
}

struct Rng {
  uint64_t s;
  explicit Rng(uint64_t seed) : s(seed * 0x9E3779B97F4A7C15ull + 1) {}
  uint32_t next() {
    s = s * 6364136223846793005ull + 1442695040888963407ull;
    return uint32_t(s >> 33);
  }
};

void put(Log* log, uint64_t cycle, const char* what, uint32_t who, const Pkt& p) {
  std::ostringstream os;
  os << "c" << cycle << " " << what << who << " val" << p.val << " tag" << p.tag;
  log->push_back(os.str());
}

// Traffic: requester i sends with probability 2/3 per cycle when it can; a
// reader pops with probability 3/4; a responder answers after 3 cycles.
struct Traffic {
  static bool send(Rng& r) { return r.next() % 3 != 0; }
  static bool take(Rng& r) { return r.next() % 4 != 0; }
  static constexpr uint64_t kLatency = 3;
};

struct Spec {
  std::string name;
  int kind;  // 0 TxArbiter, 1 TxCrossBar, 2 TxRxArbiter, 3 TxRxCrossBar
  uint32_t inputs;
  uint32_t outputs;
  ArbiterType policy;
  uint64_t gold;   // hash of the types.h unit's log
  size_t events;   // its length
};

constexpr uint64_t kCycles = 3000;


using vortex::SimContext;

class Req3 : public SimObject<Req3> {
public:
  SimPortOut<Pkt> out{this};
  SimPortIn<Pkt> rsp_in{this, 2};
  uint32_t id;
  Rng rng;
  uint32_t count = 0;
  Log* log = nullptr;
  Req3(const SimContext& c, const std::string& name, uint32_t id_) : SimObject(c, name), id(id_), rng(id_ + 1) {}

  void on_tick() {
    bool want = Traffic::send(rng);
    uint32_t addr = rng.next();
    if (!rsp_in.empty()) {
      put(log, SimPlatform::instance().cycles(), "rsp@", id, rsp_in.peek());
      rsp_in.pop();
    }
    if (want && !out.full()) {
      Pkt p;
      p.val = id * 100000 + count++;
      p.tag = count & 0xff;
      p.addr = addr;
      out.send(p, 1);
    }
  }
};

const Pkt& payload(const Pkt& p) { return p; }
template <typename R>
const Pkt& payload(const R& r) { return r.data; }

// Reads P: a Pkt, or a TxArbiter/TxCrossBar output carrying one.
template <typename P>
class Rd3 : public SimObject<Rd3<P>> {
public:
  SimPortIn<P> in{this, 2};
  SimPortOut<Pkt> rsp{this};
  uint32_t id;
  Rng rng;
  std::deque<std::pair<uint64_t, Pkt>> pending;
  bool responds = false;
  Log* log = nullptr;
  Rd3(const SimContext& c, const std::string& name, uint32_t id_) : SimObject<Rd3>(c, name), id(id_), rng(1000 + id_) {}

  void on_tick() {
    uint64_t now = SimPlatform::instance().cycles();
    bool take = Traffic::take(rng);
    if (take && !in.empty() && pending.size() < 4) {
      put(log, now, "req@", id, payload(in.peek()));
      pending.push_back({now + Traffic::kLatency, payload(in.peek())});
      in.pop();
    }
    if (!pending.empty() && pending.front().first <= now) {
      if (!responds) {
        pending.pop_front();
      } else if (!rsp.full()) {
        rsp.send(pending.front().second, 1);
        pending.pop_front();
      }
    }
  }
};

// Creates the readers after the unit, as when the hashes were recorded.
template <typename P>
std::vector<std::shared_ptr<Rd3<P>>> make_readers(const Spec& sp, Log* log, bool responds) {
  std::vector<std::shared_ptr<Rd3<P>>> rds;
  for (uint32_t o = 0; o < sp.outputs; ++o) {
    rds.push_back(Rd3<P>::Create("r" + std::to_string(o), o));
    rds.back()->log = log;
    rds.back()->responds = responds;
  }
  return rds;
}

struct SelMod {
  uint32_t n;
  uint32_t operator()(const Pkt& p) const { return p.addr % n; }
};

Log run_v3(const Spec& sp) {
  Log log;
  auto& platform = vortex::SimPlatform::instance();
  uint32_t d = (sp.inputs > 2 || sp.outputs > 2) ? 1 : 0;
  // Creation order as when the hashes were recorded: requesters, the unit, readers.
  std::vector<std::shared_ptr<Req3>> reqs;
  for (uint32_t i = 0; i < sp.inputs; ++i) {
    reqs.push_back(Req3::Create("q" + std::to_string(i), i));
    reqs.back()->log = &log;
  }
  std::shared_ptr<void> keep;
  std::shared_ptr<void> keep_rds;
  if (sp.kind == 0) {
    auto u = TxArbiter<Pkt>::Create("arb", sp.policy, sp.inputs, sp.outputs, d);
    auto rds = make_readers<TxArbiter<Pkt>::RspType>(sp, &log, false);
    for (uint32_t i = 0; i < sp.inputs; ++i) {
      reqs[i]->out.bind(&u->Inputs[i]);
    }
    for (uint32_t o = 0; o < sp.outputs; ++o) {
      u->Outputs[o].bind(&rds[o]->in);
    }
    keep = u;
    keep_rds = std::make_shared<decltype(rds)>(rds);
  } else if (sp.kind == 1) {
    auto u = TxCrossBar<Pkt>::Create("xbar", sp.policy, sp.inputs, sp.outputs, SelMod{sp.outputs}, d);
    auto rds = make_readers<TxCrossBar<Pkt>::RspType>(sp, &log, false);
    for (uint32_t i = 0; i < sp.inputs; ++i) {
      reqs[i]->out.bind(&u->Inputs[i]);
    }
    for (uint32_t o = 0; o < sp.outputs; ++o) {
      u->Outputs[o].bind(&rds[o]->in);
    }
    keep = u;
    keep_rds = std::make_shared<decltype(rds)>(rds);
  } else if (sp.kind == 2) {
    auto u = TxRxArbiter<Pkt, Pkt>::Create("txrx", sp.policy, sp.inputs, sp.outputs, d, d);
    auto rds = make_readers<Pkt>(sp, &log, true);
    for (uint32_t i = 0; i < sp.inputs; ++i) {
      reqs[i]->out.bind(&u->ReqIn[i]);
      u->RspOut[i].bind(&reqs[i]->rsp_in);
    }
    for (uint32_t o = 0; o < sp.outputs; ++o) {
      u->ReqOut[o].bind(&rds[o]->in);
      rds[o]->rsp.bind(&u->RspIn[o]);
    }
    keep = u;
    keep_rds = std::make_shared<decltype(rds)>(rds);
  } else {
    auto u = TxRxCrossBar<Pkt, Pkt>::Create("txrxx", sp.policy, sp.inputs, sp.outputs, SelMod{sp.outputs}, d, d);
    auto rds = make_readers<Pkt>(sp, &log, true);
    for (uint32_t i = 0; i < sp.inputs; ++i) {
      reqs[i]->out.bind(&u->ReqIn[i]);
      u->RspOut[i].bind(&reqs[i]->rsp_in);
    }
    for (uint32_t o = 0; o < sp.outputs; ++o) {
      u->ReqOut[o].bind(&rds[o]->in);
      rds[o]->rsp.bind(&u->RspIn[o]);
    }
    keep = u;
    keep_rds = std::make_shared<decltype(rds)>(rds);
  }
  platform.reset();
  for (uint64_t c = 0; c < kCycles; ++c) {
    platform.tick();
  }
  reqs.clear();
  keep_rds.reset();
  keep.reset();
  platform.finalize();
  return log;
}

}  // namespace

int main() {
  using P = ArbiterType;
  std::vector<Spec> specs = {
    {"TxArbiter 4:1 round robin", 0, 4, 1, P::RoundRobin, 0xe98387a1ada08c86ull, 2255},
    {"TxArbiter 4:1 priority", 0, 4, 1, P::Priority, 0x731e6c37e7b3cb51ull, 2255},
    {"TxArbiter 8:2 matrix", 0, 8, 2, P::Matrix, 0x7420915105a60b26ull, 4479},
    {"TxArbiter 3:1 greedy rr", 0, 3, 1, P::GreedyRR, 0xc0f255a1cca6caa3ull, 2254},
    {"TxArbiter 2:1 greedy matrix (delay 0)", 0, 2, 1, P::GreedyMatrix, 0xa2b637fb9304f494ull, 2254},
    {"TxArbiter 4:4 bypass", 0, 4, 4, P::RoundRobin, 0xfe222b2a4cde216full, 6865},
    {"TxCrossBar 4:2 round robin", 1, 4, 2, P::RoundRobin, 0x873a23f4be055fb1ull, 3921},
    {"TxCrossBar 4:4 priority", 1, 4, 4, P::Priority, 0xe27a96744ebc1d5full, 5737},
    {"TxCrossBar 3:2 round robin", 1, 3, 2, P::RoundRobin, 0xd3cb378dbb58fe82ull, 3729},
    {"TxCrossBar 8:4 round robin", 1, 8, 4, P::RoundRobin, 0x6ac12223dec3b976ull, 7200},
    {"TxCrossBar 2:2 round robin (delay 0)", 1, 2, 2, P::RoundRobin, 0xb480b28073de7744ull, 3360},
    {"TxCrossBar 1:1 bypass", 1, 1, 1, P::RoundRobin, 0x6079f6ec6deff968ull, 1752},
    {"TxRxArbiter 4:1 round robin", 2, 4, 1, P::RoundRobin, 0xf976dbecd90bed9full, 4507},
    {"TxRxArbiter 8:2 round robin", 2, 8, 2, P::RoundRobin, 0xb7ef9b429d371924ull, 8952},
    {"TxRxArbiter 2:1 priority (delay 0)", 2, 2, 1, P::Priority, 0x8d263fcb12e5bb1dull, 4507},
    {"TxRxArbiter 2:2 bypass", 2, 2, 2, P::RoundRobin, 0x18f03839db0c21a7ull, 6874},
    {"TxRxCrossBar 4:2 round robin", 3, 4, 2, P::RoundRobin, 0xec1b84ef1548615cull, 7830},
    {"TxRxCrossBar 8:4 round robin", 3, 8, 4, P::RoundRobin, 0x6500950316412e27ull, 14390},
    {"TxRxCrossBar 2:2 priority (delay 0)", 3, 2, 2, P::Priority, 0x28fd4e59cd66d775ull, 6704},
    {"TxRxCrossBar 1:1 bypass", 3, 1, 1, P::RoundRobin, 0x1b41dbd02a2bdc61ull, 3502},
  };
  int failures = 0;
  for (auto& sp : specs) {
    Log log = run_v3(sp);
    uint64_t h = 1469598103934665603ull;
    for (auto& e : log) {
      for (unsigned char ch : e) {
        h = (h ^ ch) * 1099511628211ull;
      }
      h = (h ^ 0xff) * 1099511628211ull;
    }
    bool ok = (h == sp.gold) && (log.size() == sp.events);
    std::printf("%-4s %-40s events=%-5zu\n", ok ? "PASS" : "FAIL", sp.name.c_str(), log.size());
    failures += !ok;
  }
  std::printf("%s: %d failing\n", failures ? "FAILED" : "PASSED", failures);
  return failures ? 1 : 0;
}
