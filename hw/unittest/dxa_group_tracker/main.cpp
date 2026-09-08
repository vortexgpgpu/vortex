#include "vl_simulator.h"
#include "VVX_dxa_group_tracker_top.h"
#include "VX_config.h"
#include <array>
#include <cstdint>
#include <cstdio>
#include <random>
#include <vector>

static uint64_t timestamp;
static unsigned checks, failures;
double sc_time_stamp() { return timestamp; }
#define CHECK(c, message) do { ++checks; if (!(c)) { ++failures; std::printf("FAIL @%llu: %s\n", (unsigned long long)timestamp, message); } } while (0)

struct Token { uint8_t wid, gid; };
constexpr unsigned depth_limit = VX_CFG_DXA_GROUP_DEPTH;
constexpr unsigned pending_limit = VX_CFG_DXA_GROUP_MAX_PENDING;
constexpr unsigned pointer_mask = 2 * depth_limit - 1;
constexpr unsigned log2ceil(unsigned n) { return n > 1 ? 1 + log2ceil((n + 1) / 2) : 0; }
constexpr unsigned pointer_width = log2ceil(depth_limit) + 1;

struct Bench {
  vl_simulator<VVX_dxa_group_tracker_top> sim;
  Bench() { reset(); }
  void clear() {
    sim->issue_valid = sim->issue_query = sim->issue_wid = 0;
    sim->commit_valid = sim->commit_wid = 0;
    sim->completion_valid = sim->completion_wid = sim->completion_gid = 0;
    sim->wq_valid = sim->wq_wid = sim->wq_n = 0;
    sim->owner_close_valid = sim->owner_close_wid = 0;
    sim->owner_start_valid = sim->owner_start_wid = 0;
    sim->query_wid = sim->query_gid = 0;
  }
  void reset() { clear(); timestamp = sim.reset(timestamp); clear(); sim->eval(); }
  void tick() { timestamp = sim.step(timestamp, 2); clear(); sim->eval(); }
  unsigned head(unsigned w) { return (sim->head_flat >> (pointer_width * w)) & pointer_mask; }
  unsigned tail(unsigned w) { return (sim->tail_flat >> (pointer_width * w)) & pointer_mask; }
  unsigned depth(unsigned w) { return (tail(w) - head(w)) & pointer_mask; }
  unsigned pending(Token t) { sim->query_wid = t.wid; sim->query_gid = t.gid; sim->eval(); return sim->query_pending; }
  bool ready(unsigned w) { sim->issue_wid = w; sim->eval(); return sim->issue_result == 0; }
  Token prepare_issue(unsigned w) {
    CHECK(ready(w), "issue unexpectedly blocked");
    Token t{uint8_t(w), uint8_t(sim->issue_gid)};
    sim->issue_query = sim->issue_valid = 1;
    return t;
  }
  Token issue(unsigned w) { auto t = prepare_issue(w); tick(); return t; }
  void prepare_complete(Token t) {
    sim->completion_valid = 1;
    sim->completion_wid = t.wid;
    sim->completion_gid = t.gid;
  }
  void complete(Token t) { prepare_complete(t); tick(); }
  void prepare_commit(unsigned w) {
    sim->commit_wid = w;
    sim->eval();
    CHECK(sim->commit_result == 0, "commit unexpectedly blocked");
    sim->commit_valid = 1;
  }
  void commit(unsigned w) { prepare_commit(w); tick(); }
  bool wait(unsigned w, unsigned n) {
    sim->wq_valid = 1; sim->wq_wid = w; sim->wq_n = n; sim->eval();
    bool pass = sim->wq_satisfied;
    CHECK(bool(sim->unlock_mask & (1 << w)) == pass, "immediate wait unlock mismatch");
    tick(); return pass;
  }
};

static void random_events(Bench &b) {
  b.reset();
  std::mt19937 random(0x5326);
  std::array<unsigned, 2> head{}, tail{}, wait_n{};
  std::array<bool, 2> open{}, waiting{};
  std::array<std::array<unsigned, depth_limit>, 2> pending{};
  for (unsigned cycle = 0; cycle < 3000; ++cycle) {
    auto committed_tail = tail;
    unsigned w = random() % 2, command = random() % 3;
    std::vector<Token> live;
    for (unsigned owner = 0; owner < 2; ++owner)
      for (unsigned g = 0; g < depth_limit; ++g)
        if (pending[owner][g]) live.push_back({uint8_t(owner), uint8_t(g)});
    if (!live.empty() && random() % 3 != 0) {
      auto done = live[random() % live.size()];
      b.prepare_complete(done);
    }
    bool issue = false, commit = false, wait = false;
    unsigned gid = tail[w] & (depth_limit - 1), n = random() % 3;
    if (!waiting[w]) {
      if (command == 0) {
        b.sim->issue_query = 1; b.sim->issue_wid = w; b.sim->eval();
        bool ready = (open[w] || ((tail[w] - head[w]) & pointer_mask) < depth_limit) && pending[w][gid] < pending_limit;
        CHECK((b.sim->issue_result == 0) == ready, "random issue readiness");
        b.sim->issue_valid = issue = ready;
      } else if (command == 1) {
        b.sim->commit_wid = w; b.sim->commit_valid = commit = true;
      } else {
        b.sim->wq_valid = wait = true; b.sim->wq_wid = w; b.sim->wq_n = n;
      }
    }
    if (issue) { ++pending[w][gid]; open[w] = true; }
    if (b.sim->completion_valid)
      --pending[b.sim->completion_wid][b.sim->completion_gid];
    if (commit && open[w]) { tail[w] = (tail[w] + 1) & pointer_mask; open[w] = false; }
    for (unsigned owner = 0; owner < 2; ++owner)
      if (head[owner] != committed_tail[owner] && pending[owner][head[owner] & (depth_limit - 1)] == 0)
        head[owner] = (head[owner] + 1) & pointer_mask;
    b.sim->eval();
    for (unsigned owner = 0; owner < 2; ++owner) {
      unsigned depth = (tail[owner] - head[owner]) & pointer_mask;
      bool wake = waiting[owner] && depth <= wait_n[owner];
      if (wait && owner == w) {
        CHECK(bool(b.sim->wq_satisfied) == (depth <= n), "random wait query");
        wake |= depth <= n;
      }
      CHECK(bool(b.sim->unlock_mask & (1 << owner)) == wake, "random wake ownership");
      if (wake) waiting[owner] = false;
      if (wait && owner == w) { waiting[owner] = depth > n; wait_n[owner] = n; }
    }
    b.tick();
    for (unsigned owner = 0; owner < 2; ++owner) {
      CHECK(b.head(owner) == head[owner] && b.tail(owner) == tail[owner], "random head/tail state");
      CHECK(bool(b.sim->obs_open & (1 << owner)) == open[owner], "random open state");
      for (unsigned g = 0; g < depth_limit; ++g)
        CHECK(b.pending({uint8_t(owner), uint8_t(g)}) == pending[owner][g], "random per-group pending state");
    }
  }
}

int main(int argc, char **argv) {
  Verilated::commandArgs(argc, argv);
  Bench b;
  b.commit(0);
  CHECK(b.tail(0) == 0 && b.wait(0, 0), "empty commit must always be a noop");

  auto a = b.issue(0);
  CHECK(b.wait(0, 0), "wait must ignore open group");
  b.complete(a);
  CHECK(b.pending(a) == 0 && (b.sim->obs_open & 1), "early completion must preserve issued boundary");
  b.commit(0);
  CHECK(b.tail(0) == 1 && b.head(0) == 0, "commit registers source-complete boundary");
  b.tick();
  CHECK(b.head(0) == 1, "early-completed commit retires next cycle");

  b.reset();
  std::vector<Token> operations;
  for (unsigned i = 0; i < pending_limit; ++i)
    operations.push_back(b.issue(0));
  a = operations[0];
  CHECK(b.pending(a) == pending_limit, "operations share one bounded counter");
  CHECK(!b.ready(0) && b.ready(1), "counter-full must be per warp");
  b.complete(operations.back()); operations.pop_back();
  CHECK(b.ready(0) && b.pending(a) == pending_limit - 1, "completion before commit returns a counter credit");
  auto e = b.issue(0);
  b.commit(0);
  CHECK(!b.wait(0, 0), "nonzero committed group must stall");
  for (auto t : operations)
    b.complete(t);
  b.prepare_complete(e); b.sim->eval();
  CHECK(b.sim->unlock_mask & 1, "final source event must wake waiter");
  b.tick();
  CHECK(b.depth(0) == 0, "completed group retires");

  b.reset();
  std::array<Token, depth_limit> groups;
  for (auto &t : groups) { t = b.issue(0); b.commit(0); }
  CHECK(b.depth(0) == depth_limit && !b.ready(0), "ring-full backpressure");
  b.commit(0);
  CHECK(b.depth(0) == depth_limit, "empty commit is noop even when full");
  auto other = b.issue(1); b.commit(1);
  const unsigned wait_n = depth_limit > 2 ? 2 : 1;
  CHECK(!b.wait(0, wait_n), "full ring exceeds wait threshold");
  for (unsigned i = depth_limit - 1; i > 0; --i)
    b.complete(groups[i]);
  CHECK(b.depth(0) == depth_limit, "younger completed groups cannot bypass old head");
  b.complete(other);
  CHECK(b.depth(1) == 0 && b.depth(0) == depth_limit, "other warp progresses while waiter blocked");
  b.prepare_complete(groups[0]); b.sim->eval();
  if (depth_limit == 2)
    CHECK(b.sim->unlock_mask & 1, "wait1 wakes at ordered depth1");
  b.tick();
  CHECK(b.depth(0) == depth_limit - 1, "retirement is one group per cycle");
  while (b.depth(0) > wait_n + 1)
    b.tick();
  if (depth_limit > 2) {
    b.sim->eval(); CHECK(b.sim->unlock_mask & 1, "wait2 wakes at ordered depth2");
    b.tick();
    CHECK(b.depth(0) == 2, "wait2 target reached");
  }
  while (b.depth(0))
    b.tick();
  a = b.issue(0); b.commit(0);
  CHECK(b.wait(0, 2) && b.wait(0, 1), "wait1/2 accept one outstanding group");
  CHECK(!b.wait(0, 0), "wait0 stalls on last group");
  b.complete(a);

  b.reset();
  for (unsigned i = 0; i < 40; ++i) {
    a = b.prepare_issue(0); b.prepare_complete(a); b.tick();
    CHECK(b.pending(a) == 0, "same-cycle issue/complete arithmetic");
    b.commit(0);
    b.tick();
    CHECK(b.head(0) == ((i + 1) & pointer_mask) && b.tail(0) == b.head(0), "pointer wrap");
  }
  a = b.issue(0); b.prepare_commit(0); b.prepare_complete(a); b.tick();
  CHECK(b.depth(0) == 1 && b.pending(a) == 0, "same-cycle commit/final complete");
  b.tick();
  CHECK(b.depth(0) == 0, "same-cycle commit/complete retires registered boundary");

  a = b.issue(0);
  b.sim->owner_close_valid = 1; b.sim->owner_close_wid = 0; b.tick();
  CHECK(!b.ready(0) && !(b.sim->drained_mask & 1), "closed owner retains live open source");
  b.complete(a);
  CHECK(b.sim->drained_mask & 1, "uncommitted completed source can drain on exit");
  b.sim->owner_start_valid = 1; b.sim->owner_start_wid = 0; b.tick();
  CHECK(b.ready(0) && b.tail(0) == 0 && b.head(0) == 0, "drained owner reinitializes");
  b.issue(0); b.commit(0); b.issue(1); b.reset();
  CHECK(b.sim->drained_mask == 3, "coordinated reset clears outstanding state");
  for (unsigned w = 0; w < 2; ++w)
    for (unsigned g = 0; g < depth_limit; ++g)
      CHECK(b.pending({uint8_t(w), uint8_t(g)}) == 0, "reset clears every pending counter");

  random_events(b);
  std::printf("checks=%u failed=%u\n%s\n", checks, failures, failures ? "FAILED" : "PASSED");
  return failures != 0;
}
