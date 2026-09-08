// Copyright © 2026
// SPDX-License-Identifier: Apache-2.0

#include "dxa/dxa_group_tracker.h"
#include <array>
#include <iostream>
#include <random>

using vortex::DxaGroupTracker;
using Token = DxaGroupTracker::OperationToken;
using Result = DxaGroupTracker::IssueResult;

static unsigned checks = 0;

static void check(bool condition, const char* message) {
  ++checks;
  if (!condition) {
    throw std::logic_error(message);
  }
}

static Token issue(DxaGroupTracker& tracker, unsigned wid) {
  Token token;
  check(tracker.issue(wid, &token) == Result::Tracked, "issue rejected");
  return token;
}

static void directed() {
  DxaGroupTracker tracker(2, 4, 3);
  tracker.commit(0);
  check(tracker.committed_depth(0) == 0 && tracker.wait_satisfied(0, 0), "empty commit created boundary");
  auto a = issue(tracker, 0);
  auto b = issue(tracker, 0);
  auto c = issue(tracker, 0);
  check(a.group_id == b.group_id && b.group_id == c.group_id, "open group identities differ");
  check(tracker.issue(0, nullptr) == Result::Backpressured, "pending counter overflowed");
  check(tracker.wait_satisfied(0, 0), "wait included uncommitted group");
  tracker.complete(0, b.group_id);
  auto d = issue(tracker, 0);
  check(d.group_id == a.group_id && tracker.open_remaining(0) == 3, "early completion did not return credit");
  tracker.complete(0, a.group_id);
  tracker.complete(0, c.group_id);
  tracker.complete(0, d.group_id);
  check(tracker.open_valid(0) && tracker.open_remaining(0) == 0, "early completion lost open boundary");
  tracker.commit(0);
  check(tracker.committed_depth(0) == 1, "issued but completed group treated as empty");
  tracker.tick();
  check(tracker.wait_satisfied(0, 0), "completed group did not retire");

  std::array<Token, 4> groups;
  for (auto& token : groups) {
    token = issue(tracker, 0);
    tracker.commit(0);
  }
  check(tracker.committed_depth(0) == 4, "ring never became full");
  check(tracker.issue(0, nullptr) == Result::Backpressured, "full ring accepted issue");
  tracker.commit(0);
  check(tracker.committed_depth(0) == 4, "empty commit changed full ring");
  auto independent = issue(tracker, 1);
  tracker.commit(1);
  tracker.complete(1, independent.group_id);
  tracker.tick();
  check(tracker.wait_satisfied(1, 0) && !tracker.wait_satisfied(0, 2), "warps are not independent");
  tracker.complete(0, groups[2].group_id);
  tracker.complete(0, groups[1].group_id);
  tracker.tick();
  check(tracker.committed_depth(0) == 4, "out-of-order completion skipped head");
  tracker.complete(0, groups[0].group_id);
  tracker.tick();
  check(tracker.committed_depth(0) == 3, "more than one head retired in a tick");
  tracker.tick();
  check(tracker.wait_satisfied(0, 2) && !tracker.wait_satisfied(0, 1), "wait 2 frontier incorrect");
  tracker.tick();
  check(tracker.wait_satisfied(0, 1) && !tracker.wait_satisfied(0, 0), "wait 1 frontier incorrect");
  tracker.complete(0, groups[3].group_id);
  tracker.tick();
  check(tracker.wait_satisfied(0, 0), "wait 0 did not drain");

  for (unsigned i = 0; i < 1024; ++i) {
    auto token = issue(tracker, 0);
    if (i & 1) {
      tracker.commit(0);
      tracker.complete(0, token.group_id);
    } else {
      tracker.complete(0, token.group_id);
      tracker.commit(0);
    }
    tracker.tick();
    check(tracker.drained(0), "wrap or commit/complete ordering failed");
  }

  auto live = issue(tracker, 0);
  tracker.close_owner(0);
  check(!tracker.reinitialize_owner(0), "reused owner with live source");
  check(tracker.issue(0, nullptr) == Result::OwnerClosed, "closed owner accepted issue");
  tracker.complete(0, live.group_id);
  check(tracker.reinitialize_owner(0), "source-complete open group prevented owner reuse");
  check(!tracker.open_valid(0) && tracker.open_gid(0) == 0, "owner reuse retained old open group");
  bool rejected = false;
  try {
    tracker.complete(0, 0);
  } catch (const std::logic_error&) {
    rejected = true;
  }
  check(rejected, "completion underflow not detected");
  issue(tracker, 0);
  tracker.reset();
  check(tracker.drained(0) && !tracker.open_valid(0), "reset retained pending state");
}

static void randomized() {
  constexpr unsigned warps = 4, depth = 8, limit = 8;
  DxaGroupTracker tracker(warps, depth, limit);
  struct Oracle {
    unsigned head = 0, tail = 0, open_pending = 0;
    bool open = false;
    std::vector<unsigned> committed;
  };
  std::array<Oracle, warps> oracle;
  std::mt19937 random(0x583267);
  for (unsigned cycle = 0; cycle < 20000; ++cycle) {
    unsigned wid = random() % warps;
    auto& o = oracle[wid];
    switch (random() % 3) {
    case 0: {
      bool ready = (o.open || o.committed.size() < depth) && o.open_pending < limit;
      Token token;
      auto result = tracker.issue(wid, &token);
      check((result == Result::Tracked) == ready, "random issue readiness mismatch");
      if (ready) {
        check(token.group_id == o.tail % depth, "random issue gid mismatch");
        o.open = true;
        ++o.open_pending;
      }
      break;
    }
    case 1:
      tracker.commit(wid);
      if (o.open) {
        o.committed.push_back(o.open_pending);
        ++o.tail;
        o.open = false;
        o.open_pending = 0;
      }
      break;
    case 2: {
      unsigned index = random() % (o.committed.size() + 1);
      if (index < o.committed.size() && o.committed[index]) {
        tracker.complete(wid, (o.head + index) % depth);
        --o.committed[index];
      } else if (index == o.committed.size() && o.open_pending) {
        tracker.complete(wid, o.tail % depth);
        --o.open_pending;
      }
      break;
    }
    }
    tracker.tick();
    for (unsigned w = 0; w < warps; ++w) {
      auto& expected = oracle[w];
      if (!expected.committed.empty() && expected.committed.front() == 0) {
        expected.committed.erase(expected.committed.begin());
        ++expected.head;
      }
      check(tracker.committed_depth(w) == expected.committed.size(), "random depth mismatch");
      check(tracker.open_remaining(w) == expected.open_pending, "random open count mismatch");
      for (unsigned n = 0; n < 3; ++n) {
        check(tracker.wait_satisfied(w, n) == (expected.committed.size() <= n), "random wait mismatch");
      }
    }
  }
}

int main() {
  directed();
  randomized();
  std::cout << "DXA simple group tests: checks=" << checks << " PASSED\n";
}
