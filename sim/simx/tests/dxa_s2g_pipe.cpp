// Copyright © 2026
// SPDX-License-Identifier: Apache-2.0

#include "dxa/dxa_s2g_pipe.h"
#include <array>
#include <iostream>
#include <random>

using vortex::DxaS2gPipe;

static void run(unsigned slots, unsigned credits, unsigned word_bytes) {
  constexpr unsigned lines = 48, line_bytes = 64;
  DxaS2gPipe pipe(slots, credits, word_bytes, line_bytes);
  std::vector<uint8_t> source(32768), output(lines * line_bytes + 64, 0xd7), expected = output;
  std::vector<DxaS2gPipe::Line> work;
  std::vector<DxaS2gPipe::Read> pending;
  std::mt19937 random(54321);
  for (unsigned i = 0; i < source.size(); ++i) {
    source[i] = uint8_t(i * 23 + i / 97);
  }
  for (unsigned i = 0; i < lines; ++i) {
    DxaS2gPipe::Line line{32 + i * line_bytes, 256 + i * 193 + i % 13, i % 7, i % 9 ? 57u : 0u, i + 1 == lines};
    work.push_back(line);
    std::memcpy(expected.data() + line.destination + line.offset, source.data() + line.source, line.length);
  }
  unsigned issued = 0, reads = 0, reordered = 0;
  bool overwritten = false;
  for (unsigned cycle = 0; cycle < 100000; ++cycle) {
    if (!pending.empty() && (random() % 5 == 0 || pending.size() == credits)) {
      unsigned index = random() % pending.size();
      auto read = pending[index];
      reordered += index != 0;
      pipe.complete(read.tag, source.data() + read.address, word_bytes);
      pending.erase(pending.begin() + index);
    }
    if (pipe.source_done() && !overwritten) {
      std::fill(source.begin(), source.end(), 0xa5);
      overwritten = true;
    }
    DxaS2gPipe::Line line;
    const std::vector<uint8_t>* payload;
    if (pipe.peek_store(&line, &payload)) {
      const auto before = *payload;
      DxaS2gPipe::Line again;
      const std::vector<uint8_t>* again_payload;
      if (!pipe.peek_store(&again, &again_payload) || again.destination != line.destination || *again_payload != before) {
        throw std::logic_error("store request changed without acceptance");
      }
      if (random() % 3 == 0) {
        std::memcpy(output.data() + line.destination + line.offset, payload->data() + line.offset, line.length);
        pipe.accept_store();
      }
    }
    DxaS2gPipe::Read read;
    if (pipe.peek_read(&read) && random() % 4 != 0) {
      pending.push_back(read);
      pipe.accept_read();
      ++reads;
    }
    if (issued < work.size() && pipe.push(work[issued])) {
      ++issued;
    }
    if (overwritten && pipe.empty()) {
      if (output != expected || !pending.empty() || issued != work.size()) {
        throw std::logic_error("S2G pipe output/canary/source-lifetime mismatch");
      }
      if (credits > 1 && slots > 1 && (!reordered || pipe.peak_pending() < 2)) {
        throw std::logic_error("S2G pipe did not exercise multiple outstanding words");
      }
      std::cout << "slots=" << slots << " credits=" << credits << " word=" << word_bytes
                << " reads=" << reads << " peak=" << pipe.peak_pending() << " reordered=" << reordered << " PASSED\n";
      return;
    }
  }
  throw std::logic_error("S2G pipe timeout");
}

int main() {
  run(1, 1, 16);
  run(3, 4, 4);
  run(4, 4, 16);
  run(8, 8, 16);
  run(4, 4, 256);
  DxaS2gPipe zero(4, 4, 16, 64);
  zero.reset(true);
  if (!zero.source_done() || !zero.empty()) {
    throw std::logic_error("zero-length pipe does not drain");
  }
}
