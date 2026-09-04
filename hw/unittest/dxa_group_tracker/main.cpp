#include "vl_simulator.h"
#include "VVX_dxa_group_tracker_top.h"
#include "dxa_group_tracker.h"

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

  Op issue_and_complete(int wid) {
    sim->issue_query = 1;
    sim->issue_wid = wid;
    sim->eval();
    CHECK(sim->issue_result == 0,
          "warp %d same-cycle issue unexpectedly blocked", wid);
    Op op{uint8_t(wid), uint8_t((sim->epoch_flat >> (wid * 2)) & 3),
          tail(wid), uint8_t(sim->issue_op_id)};
    sim->issue_valid = 1;
    sim->completion_valid = 1;
    sim->completion_wid = op.wid;
    sim->completion_epoch = op.epoch;
    sim->completion_seq = op.seq;
    sim->completion_op = op.id;
    sim->eval();
    CHECK(sim->completion_ready,
          "same-cycle completion transport backpressured");
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

static void check_functional_model_edges() {
  // The RTL wrapper intentionally uses a tiny two-warp/four-context shape.
  // Exercise the model with a larger per-warp partition so a single warp can
  // fill all four boundary rows without needing an artificial second port.
  vortex::DxaGroupTracker model(/*warps=*/1, /*contexts=*/8,
                                /*ring=*/4, /*generation_bits=*/4);
  std::array<vortex::DxaGroupTracker::OperationToken, 4> pending{};
  for (auto& token : pending) {
    CHECK(model.issue(0, &token)
              == vortex::DxaGroupTracker::IssueResult::Tracked,
          "model issue unexpectedly blocked while filling the ring");
    CHECK(model.commit(0)
              == vortex::DxaGroupTracker::CommitResult::Accepted,
          "model commit unexpectedly blocked while filling the ring");
  }
  const uint8_t tail_before = model.sealed_tail(0);
  CHECK(model.committed_depth(0) == 4,
        "model did not retain four pending boundaries");
  CHECK(model.commit(0)
            == vortex::DxaGroupTracker::CommitResult::Accepted,
        "empty commit must remain a trivially complete no-op at ring full");
  CHECK(model.sealed_tail(0) == tail_before,
        "full-ring empty commit advanced the sequence/overwrote a row");
  for (const auto& token : pending) {
    CHECK(model.complete(0, token.epoch, token.group_seq, token.op_id)
              == vortex::DxaGroupTracker::CompletionResult::Applied,
          "model completion failed while draining full ring");
  }
  for (int i = 0; i < 4; ++i)
    model.tick();
  CHECK(model.drained(0), "model did not drain completed boundaries");

  model.reset();
  vortex::DxaGroupTracker::OperationToken old;
  CHECK(model.issue(0, &old)
            == vortex::DxaGroupTracker::IssueResult::Tracked,
        "model epoch setup issue failed");
  CHECK(model.complete(0, old.epoch, old.group_seq, old.op_id)
            == vortex::DxaGroupTracker::CompletionResult::Applied,
        "model epoch setup completion failed");
  const uint32_t old_epoch = model.epoch(0);
  CHECK(model.advance_epoch(0),
        "completed uncommitted open group was not epoch-drainable");
  CHECK(model.epoch(0) == ((old_epoch + 1) & 3),
        "model epoch did not advance after open-group discard");
  vortex::DxaGroupTracker::OperationToken fresh;
  CHECK(model.issue(0, &fresh)
            == vortex::DxaGroupTracker::IssueResult::Tracked,
        "model rejected first issue in a fresh epoch");
  CHECK(fresh.epoch != old.epoch && fresh.group_seq == 0,
        "fresh epoch retained stale open-group identity");

  // An uncommitted group is intentionally invisible to wait.read.  Once it
  // is sealed, the same three heterogeneous operations form one boundary and
  // the N threshold is evaluated against groups (not operation count).
  model.reset();
  std::array<vortex::DxaGroupTracker::OperationToken, 3> triple{};
  for (auto& token : triple)
    CHECK(model.issue(0, &token)
              == vortex::DxaGroupTracker::IssueResult::Tracked,
          "model rejected a second/third operation in an open group");
  const uint8_t open_snapshot = model.wait_snapshot(0);
  CHECK(model.wait_satisfied(0, 0, open_snapshot),
        "wait.read observed an uncommitted open group");
  CHECK(model.commit(0) == vortex::DxaGroupTracker::CommitResult::Accepted,
        "three-operation group did not commit");
  const uint8_t first_seq = triple[0].group_seq;
  CHECK(first_seq != model.wait_snapshot(0),
        "commit did not advance the group boundary");
  CHECK(!model.wait_satisfied(0, 0, model.wait_snapshot(0)),
        "wait<0> passed while the committed group was pending");
  CHECK(model.wait_satisfied(0, 1, model.wait_snapshot(0)),
        "wait<1> did not retain the newest committed group");
  // Complete out of order; FIFO retirement must still wait for all three.
  CHECK(model.complete(0, triple[2].epoch, triple[2].group_seq, triple[2].op_id)
            == vortex::DxaGroupTracker::CompletionResult::Applied,
        "third operation completion failed");
  model.tick();
  CHECK(!model.wait_satisfied(0, 0, model.wait_snapshot(0)),
        "out-of-order completion retired a partial group");
  CHECK(model.complete(0, triple[0].epoch, triple[0].group_seq, triple[0].op_id)
            == vortex::DxaGroupTracker::CompletionResult::Applied,
        "first operation completion failed");
  CHECK(model.complete(0, triple[1].epoch, triple[1].group_seq, triple[1].op_id)
            == vortex::DxaGroupTracker::CompletionResult::Applied,
        "second operation completion failed");
  model.tick();
  CHECK(model.wait_satisfied(0, 0, model.wait_snapshot(0)),
        "wait<0> remained blocked after the complete group retired");

  // Reset is a hard lifetime boundary: an old token must not complete a new
  // operation even when the physical slot is reused from sequence zero.
  model.reset();
  vortex::DxaGroupTracker::OperationToken before_reset;
  CHECK(model.issue(0, &before_reset)
            == vortex::DxaGroupTracker::IssueResult::Tracked,
        "reset setup issue failed");
  model.reset();
  vortex::DxaGroupTracker::OperationToken after_reset;
  CHECK(model.issue(0, &after_reset)
            == vortex::DxaGroupTracker::IssueResult::Tracked,
        "post-reset issue failed");
  CHECK(model.complete(0, before_reset.epoch, before_reset.group_seq,
                       before_reset.op_id)
            != vortex::DxaGroupTracker::CompletionResult::Applied,
        "pre-reset token mutated a post-reset context");

  // Source completion is independent of the commit boundary.  Completing an
  // open operation first releases its source lifetime, but the completed
  // group remains a real boundary until commit seals it.
  model.reset();
  vortex::DxaGroupTracker::OperationToken early;
  CHECK(model.issue(0, &early)
            == vortex::DxaGroupTracker::IssueResult::Tracked,
        "early-completion setup issue failed");
  CHECK(model.complete(0, early.epoch, early.group_seq, early.op_id)
            == vortex::DxaGroupTracker::CompletionResult::Applied,
        "completion before commit was rejected");
  CHECK(model.open_remaining(0) == 0 && model.committed_depth(0) == 0,
        "completion before commit changed the sealed depth incorrectly");
  CHECK(model.commit(0) == vortex::DxaGroupTracker::CommitResult::Accepted,
        "completed open group did not commit");
  CHECK(model.committed_depth(0) == 1,
        "completed open group disappeared before commit");
  model.tick();
  CHECK(model.drained(0), "completed group did not drain after commit");

  // A wait snapshots a boundary, then retains the newest N groups while the
  // older prefix retires.  This is the observable distinction between
  // wait<1> and a conservative wait-for-all implementation.
  model.reset();
  std::array<vortex::DxaGroupTracker::OperationToken, 3> staged{};
  for (auto& token : staged) {
    CHECK(model.issue(0, &token)
              == vortex::DxaGroupTracker::IssueResult::Tracked,
          "three-group wait setup issue failed");
    CHECK(model.commit(0) == vortex::DxaGroupTracker::CommitResult::Accepted,
          "three-group wait setup commit failed");
  }
  const uint8_t three_snapshot = model.wait_snapshot(0);
  CHECK(!model.wait_satisfied(0, 1, three_snapshot),
        "wait<1> passed with three incomplete committed groups");
  CHECK(model.complete(0, staged[0].epoch, staged[0].group_seq,
                       staged[0].op_id)
            == vortex::DxaGroupTracker::CompletionResult::Applied,
        "oldest wait group completion failed");
  model.tick();
  CHECK(!model.wait_satisfied(0, 1, three_snapshot),
        "wait<1> passed while two committed groups remained");
  CHECK(model.complete(0, staged[1].epoch, staged[1].group_seq,
                       staged[1].op_id)
            == vortex::DxaGroupTracker::CompletionResult::Applied,
        "middle wait group completion failed");
  model.tick();
  CHECK(model.wait_satisfied(0, 1, three_snapshot),
        "wait<1> did not pass with exactly one newest group retained");
  CHECK(!model.wait_satisfied(0, 0, three_snapshot),
        "wait<0> passed while the newest group remained pending");
  CHECK(model.complete(0, staged[2].epoch, staged[2].group_seq,
                       staged[2].op_id)
            == vortex::DxaGroupTracker::CompletionResult::Applied,
        "newest wait group completion failed");
  model.tick();
  CHECK(model.wait_satisfied(0, 0, three_snapshot),
        "wait<0> did not pass after all three groups retired");

  // Ring pressure is local to an issuer.  A full warp-0 boundary ring must
  // not consume warp-1's statically owned context or boundary capacity.
  vortex::DxaGroupTracker two_warp(/*warps=*/2, /*contexts=*/8,
                                   /*ring=*/4, /*generation_bits=*/4);
  std::array<vortex::DxaGroupTracker::OperationToken, 4> ring_ops{};
  for (auto& token : ring_ops) {
    CHECK(two_warp.issue(0, &token)
              == vortex::DxaGroupTracker::IssueResult::Tracked,
          "warp-0 ring fill issue failed");
    CHECK(two_warp.commit(0) == vortex::DxaGroupTracker::CommitResult::Accepted,
          "warp-0 ring fill commit failed");
  }
  vortex::DxaGroupTracker::OperationToken other_warp;
  CHECK(two_warp.issue(1, &other_warp)
            == vortex::DxaGroupTracker::IssueResult::Tracked,
        "warp-1 was blocked by warp-0 ring pressure");
  CHECK(two_warp.issue(0, nullptr)
            == vortex::DxaGroupTracker::IssueResult::Backpressured,
        "full warp-0 ring did not backpressure a new group");
  CHECK(two_warp.complete(0, ring_ops[0].epoch, ring_ops[0].group_seq,
                         ring_ops[0].op_id)
            == vortex::DxaGroupTracker::CompletionResult::Applied,
        "warp-0 ring head completion failed");
  two_warp.tick();
  CHECK(two_warp.issue(0, nullptr)
            == vortex::DxaGroupTracker::IssueResult::Tracked,
        "warp-0 did not reopen after one boundary retired");
  CHECK(two_warp.complete(1, other_warp.epoch, other_warp.group_seq,
                         other_warp.op_id)
            == vortex::DxaGroupTracker::CompletionResult::Applied,
        "warp-1 independent operation completion failed");
}

int main(int argc, char **argv) {
  Verilated::commandArgs(argc, argv);
  check_functional_model_edges();
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
    std::printf("[2b] zero-latency issue/completion is exact-once\n");
    Op instant = b.issue_and_complete(0);
    CHECK(b.open(0) == 0,
          "same-cycle completion left an open operation pending");
    b.commit(0);
    CHECK(b.head(0) == b.tail(0),
          "same-cycle completed group did not retire at commit");
    const uint32_t stale_before = uint32_t(b.sim->stale_flat);
    b.complete(instant);
    CHECK(uint32_t(b.sim->duplicate_flat) != 0
          || uint32_t(b.sim->stale_flat) != stale_before,
          "same-cycle token was not protected against a duplicate");
  }

  {
    b.reset();
    std::printf("[3] statically partitioned contexts backpressure only owner warp\n");
    std::array<Op, 4> pool;
    pool[0] = b.issue(0);
    pool[1] = b.issue(1);
    pool[2] = b.issue(0);
    pool[3] = b.issue(1);
    b.sim->issue_query = 1;
    b.sim->issue_wid = 0;
    b.sim->eval();
    CHECK(b.sim->issue_result == 1, "warp0 context partition did not backpressure");
    const uint8_t live_before = b.sim->context_live;
    b.tick();
    CHECK(b.sim->context_live == live_before,
          "backpressured issue changed physical allocation state");
    CHECK(b.sim->context_stalls == 1, "pool stall was not counted");
    b.sim->issue_wid = 1;
    b.sim->eval();
    CHECK(b.sim->issue_result == 1, "warp1 context partition did not backpressure");
    b.complete(pool[1]);
    // Freeing warp1's slot must not make warp0's issue succeed.
    b.sim->issue_wid = 0;
    b.sim->eval();
    CHECK(b.sim->issue_result == 1,
          "warp0 borrowed a context from another warp");
    Op recycled = b.issue(1);
    CHECK(recycled.id != pool[1].id,
          "recycled context did not change generation");
    b.complete(pool[0]);
    b.complete(pool[2]);
    b.complete(pool[3]);
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
    // The unit configuration has two operation contexts per warp. Reclaim
    // one old operation before opening the next group, while leaving the
    // other old operation pending to test FIFO retirement.
    b.complete(w0g0b);
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
