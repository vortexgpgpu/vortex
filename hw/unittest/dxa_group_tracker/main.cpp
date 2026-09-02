#include "vl_simulator.h"
#include "VVX_dxa_group_tracker_top.h"

#include <array>
#include <cstdint>
#include <cstdio>

using Device = VVX_dxa_group_tracker_top;

static uint64_t timestamp = 0;
static int checks = 0;
static int failures = 0;

double sc_time_stamp() { return timestamp; }

#define CHECK(cond, ...) do {                                      \
  ++checks;                                                        \
  if (!(cond)) {                                                   \
    ++failures;                                                    \
    std::printf("FAILED t=%llu: ", (unsigned long long)timestamp); \
    std::printf(__VA_ARGS__);                                      \
    std::printf("\n");                                            \
  }                                                               \
} while (0)

struct Op {
  uint8_t wid;
  uint8_t epoch;
  uint8_t seq;
  uint8_t id;
};

struct Bench {
  vl_simulator<Device> sim;

  Bench() {
    clear_inputs();
    timestamp = sim.reset(timestamp);
  }

  void reset() {
    clear_inputs();
    timestamp = sim.reset(timestamp);
    clear_inputs();
    sim->eval();
  }

  void clear_inputs() {
    sim->assert_on_drop = 0;
    sim->issue_valid = 0;
    sim->issue_query = 0;
    sim->issue_wid = 0;
    sim->commit_valid = 0;
    sim->commit_wid = 0;
    sim->completion_valid = 0;
    sim->completion_wid = 0;
    sim->completion_epoch = 0;
    sim->completion_seq = 0;
    sim->completion_op = 0;
    sim->wq_valid = 0;
    sim->wq_wid = 0;
    sim->wq_n = 0;
    sim->poison_valid = 0;
    sim->poison_wid = 0;
    sim->epoch_adv_valid = 0;
    sim->epoch_adv_wid = 0;
    sim->sticky_clear_valid = 0;
    sim->sticky_clear_wid = 0;
    sim->sticky_clear_mask = 0;
    sim->cmp_n = 0;
    sim->cmp_snapshot = 0;
    sim->cmp_head = 0;
    sim->cmp_ring_depth = 4;
  }

  void tick() {
    timestamp = sim.step(timestamp, 2);
    clear_inputs();
    sim->eval();
  }

  uint8_t tail(int wid) {
    return (sim->sealed_tail_flat >> (wid * 8)) & 0xff;
  }

  uint8_t head(int wid) {
    return (sim->read_head_flat >> (wid * 8)) & 0xff;
  }

  uint8_t open(int wid) {
    return (sim->open_ops_flat >> (wid * 3)) & 0x7;
  }

  Op issue(int wid) {
    sim->issue_query = 1;
    sim->issue_wid = wid;
    sim->eval();
    CHECK(sim->issue_result == 0, "warp %d issue unexpectedly blocked", wid);
    Op op{uint8_t(wid), uint8_t((sim->epoch_flat >> (wid * 2)) & 3),
          tail(wid), uint8_t(sim->issue_op_id)};
    sim->issue_valid = 1;
    tick();
    return op;
  }

  void commit(int wid) {
    sim->commit_wid = wid;
    sim->eval();
    CHECK(sim->commit_result == 0, "warp %d commit unexpectedly blocked", wid);
    sim->commit_valid = 1;
    tick();
  }

  void complete(const Op &op) {
    sim->completion_valid = 1;
    sim->completion_wid = op.wid;
    sim->completion_epoch = op.epoch;
    sim->completion_seq = op.seq;
    sim->completion_op = op.id;
    sim->eval();
    CHECK(sim->completion_ready, "completion transport backpressured");
    tick();
  }

  void complete_and_commit(const Op &op) {
    sim->completion_valid = 1;
    sim->completion_wid = op.wid;
    sim->completion_epoch = op.epoch;
    sim->completion_seq = op.seq;
    sim->completion_op = op.id;
    sim->commit_valid = 1;
    sim->commit_wid = op.wid;
    sim->eval();
    CHECK(sim->completion_ready, "completion transport backpressured");
    CHECK(sim->commit_result == 0, "same-cycle commit unexpectedly blocked");
    tick();
  }

  bool query(int wid, int n) {
    sim->wq_valid = 1;
    sim->wq_wid = wid;
    sim->wq_n = n;
    sim->eval();
    return sim->wq_satisfied;
  }
};

int main(int argc, char **argv) {
  Verilated::commandArgs(argc, argv);
  Bench b;
  {
    std::printf("[1] completion+commit snapshots the decremented open count\n");
    Op op = b.issue(0);
    b.complete_and_commit(op);
    CHECK(b.open(0) == 0, "same-cycle completion was not subtracted");
    CHECK(b.head(0) == 1 && b.tail(0) == 1,
          "same-cycle zero boundary did not retire");

    const uint8_t before = b.head(0);
    b.commit(0);
    CHECK(b.head(0) == uint8_t(before + 1)
          && b.tail(0) == uint8_t(before + 1),
          "empty group did not seal and retire in its commit cycle");
  }

  {
    b.reset();
    std::printf("[2] one completed boundary retires per warp per cycle\n");
    Op blocker = b.issue(0);
    b.commit(0);
    b.commit(0);
    b.commit(0);
    b.commit(0);
    CHECK(b.head(0) == 0 && b.tail(0) == 4,
          "younger empty groups passed the incomplete head");
    b.complete(blocker);
    CHECK(b.head(0) == 1, "head group did not retire on completion");
    b.tick();
    CHECK(b.head(0) == 2, "tracker retired other than one row in cycle 1");
    b.tick();
    CHECK(b.head(0) == 3, "tracker retired other than one row in cycle 2");
    b.tick();
    CHECK(b.head(0) == 4, "tracker did not drain final empty row");
  }

  {
    b.reset();
    std::printf("[3] shared P-context pool backpressures without drop\n");
    std::array<Op, 4> pool;
    pool[0] = b.issue(0);
    pool[1] = b.issue(1);
    pool[2] = b.issue(0);
    pool[3] = b.issue(1);
    b.sim->issue_query = 1;
    b.sim->issue_wid = 0;
    b.sim->eval();
    CHECK(b.sim->issue_result == 1, "full physical pool did not backpressure");
    const uint8_t live_before = b.sim->context_live;
    b.tick();
    CHECK(b.sim->context_live == live_before,
          "backpressured issue changed physical allocation state");
    CHECK(b.sim->context_stalls == 1, "pool stall was not counted");
    b.complete(pool[1]);
    Op recycled = b.issue(0);
    CHECK(recycled.id != pool[1].id,
          "recycled context did not change generation");
    for (const auto &op : pool) {
      if (op.id != pool[1].id)
        b.complete(op);
    }
    b.complete(recycled);
  }

  {
    b.reset();
    std::printf("[4] two issuer streams retire independently under OOO completion\n");
    Op w0g0a = b.issue(0);
    Op w0g0b = b.issue(0);
    b.commit(0);
    Op w1g0 = b.issue(1);
    b.commit(1);
    Op w0g1 = b.issue(0);
    b.commit(0);

    CHECK(!b.query(0, 1), "warp0 wait<1> released before old group completed");
    b.tick();
    CHECK(!b.query(1, 0), "warp1 wait<0> released before its completion");
    b.tick();

    b.complete(w0g1);
    CHECK(b.head(0) + 2 == b.tail(0), "newer warp0 group retired past older group");
    b.complete(w1g0);
    b.tick();
    CHECK(b.head(1) == b.tail(1), "warp1 did not retire independently");
    CHECK(b.head(0) + 2 == b.tail(0), "warp1 completion disturbed warp0");

    b.complete(w0g0b);
    b.tick();
    CHECK(b.head(0) + 2 == b.tail(0), "partial old group released wait early");
    b.complete(w0g0a);
    CHECK(b.head(0) + 1 == b.tail(0),
          "old group did not retire first (head=%u tail=%u)", b.head(0), b.tail(0));
    CHECK((b.sim->unlock_mask & 1) != 0,
          "parked wait<1> did not unlock (head=%u tail=%u)", b.head(0), b.tail(0));
    b.tick();
    CHECK(b.head(0) == b.tail(0), "completed newer group did not retire next");
  }

  {
    b.reset();
    std::printf("[5] modulo comparator treats a passed snapshot as satisfied\n");
    b.sim->cmp_n = 0;
    b.sim->cmp_snapshot = 10;
    b.sim->cmp_head = 11;
    b.sim->cmp_ring_depth = 4;
    b.sim->eval();
    CHECK(b.sim->cmp_satisfied,
          "snapshot behind head was mistaken for 255 outstanding groups");
    b.sim->cmp_snapshot = 14;
    b.sim->cmp_head = 11;
    b.sim->eval();
    CHECK(!b.sim->cmp_satisfied, "three live groups unexpectedly satisfied wait<0>");
    b.sim->cmp_n = 3;
    b.sim->eval();
    CHECK(b.sim->cmp_satisfied, "wait<3> rejected exactly three live groups");
  }

  {
    b.reset();
    std::printf("[6] four-bit generation distinguishes 15 intervening reuses\n");
    Op stale = b.issue(0);
    b.complete(stale);
    b.commit(0);
    for (int reuse = 0; reuse < 15; ++reuse) {
      Op current = b.issue(0);
      CHECK(current.id != stale.id,
            "context generation aliased before 16 reuses at reuse=%d", reuse + 1);
      b.complete(current);
      b.commit(0);
    }
    const uint32_t stale_before = uint32_t(b.sim->stale_flat);
    b.complete(stale);
    CHECK(uint32_t(b.sim->stale_flat) == stale_before + 1,
          "old generation was not classified stale");
  }

  {
    b.reset();
    std::printf("[7] wid reuse waits for drain and advances the owner epoch\n");
    Op old = b.issue(0);
    const uint8_t epoch_before = uint8_t(b.sim->epoch_flat & 3);

    // The scheduler may request a new lifetime only after drained_mask rises.
    b.sim->epoch_adv_valid = 1;
    b.sim->epoch_adv_wid = 0;
    b.tick();
    CHECK(uint8_t(b.sim->epoch_flat & 3) == epoch_before,
          "epoch advanced while an S2G operation was live");

    b.complete(old);
    CHECK((b.sim->drained_mask & 1) != 0,
          "completed open group did not make the owner drainable");
    b.sim->poison_valid = 1;
    b.sim->poison_wid = 0;
    b.tick();
    b.sim->issue_query = 1;
    b.sim->issue_wid = 0;
    b.sim->eval();
    CHECK(b.sim->issue_result == 2,
          "retiring owner accepted a new S2G operation");
    b.tick();

    b.sim->epoch_adv_valid = 1;
    b.sim->epoch_adv_wid = 0;
    b.tick();
    const uint8_t epoch_after = uint8_t(b.sim->epoch_flat & 3);
    CHECK(epoch_after == uint8_t((epoch_before + 1) & 3),
          "drained wid reuse did not advance epoch");

    const uint32_t stale_before = uint32_t(b.sim->stale_flat);
    b.complete(old);
    CHECK(uint32_t(b.sim->stale_flat) == stale_before + 1,
          "late completion from the old CTA epoch was not rejected");
    Op fresh = b.issue(0);
    CHECK(fresh.epoch == epoch_after,
          "new CTA operation did not inherit the advanced epoch");
    b.complete(fresh);
  }

  std::printf("%d checks, %d failed\n", checks, failures);
  if (failures == 0) {
    std::printf("PASSED!\n");
    return 0;
  }
  return 1;
}
