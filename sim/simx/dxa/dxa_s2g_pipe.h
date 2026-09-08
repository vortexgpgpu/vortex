// Copyright © 2026
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <vector>

namespace vortex {

class DxaS2gPipe {
public:
  struct Line {
    uint64_t destination;
    uint64_t source;
    uint32_t offset;
    uint32_t length;
    bool last;
  };
  struct Read { uint64_t address; uint32_t tag; };

  DxaS2gPipe(uint32_t slots, uint32_t credits, uint32_t word_bytes, uint32_t line_bytes)
      : slots_(slots), credits_(credits), word_bytes_(word_bytes), line_bytes_(line_bytes)
      , max_words_(word_bytes ? (line_bytes + 2 * word_bytes - 2) / word_bytes : 0) {
    if (!slots || !credits || !word_bytes || !line_bytes || (word_bytes & (word_bytes - 1))) {
      throw std::invalid_argument("invalid S2G pipe dimensions");
    }
    reset();
  }

  void reset(bool zero_length = false) {
    for (auto& slot : slots_) {
      slot = Slot{};
    }
    pending_ = peak_pending_ = 0;
    held_read_ = held_store_ = slots_.size();
    last_seen_ = zero_length;
  }

  bool push(const Line& line) {
    if (last_seen_) {
      throw std::logic_error("S2G token after final token");
    }
    if (line.length == 0) {
      last_seen_ = line.last;
      return true;
    }
    if (line.offset + line.length > line_bytes_) {
      throw std::logic_error("S2G token exceeds destination line");
    }
    auto slot = std::find_if(slots_.begin(), slots_.end(), [](const Slot& s) { return !s.live; });
    if (slot == slots_.end()) {
      return false;
    }
    *slot = Slot{};
    slot->live = true;
    slot->line = line;
    slot->words = ((line.source & (word_bytes_ - 1)) + line.length + word_bytes_ - 1) / word_bytes_;
    slot->seen.resize(slot->words);
    slot->data.resize(line_bytes_);
    last_seen_ = line.last;
    return true;
  }

  bool peek_read(Read* read) {
    if (pending_ == credits_) {
      return false;
    }
    if (held_read_ == slots_.size()) {
      for (uint32_t i = 0; i < slots_.size(); ++i) {
        if (slots_[i].live && slots_[i].issued < slots_[i].words) {
          held_read_ = i;
          break;
        }
      }
    }
    if (held_read_ == slots_.size()) {
      return false;
    }
    const auto& slot = slots_[held_read_];
    read->address = (slot.line.source & ~uint64_t(word_bytes_ - 1)) + uint64_t(slot.issued) * word_bytes_;
    read->tag = held_read_ * max_words_ + slot.issued;
    return true;
  }

  void accept_read() {
    if (held_read_ == slots_.size() || pending_ == credits_) {
      throw std::logic_error("S2G read accepted without a valid request");
    }
    ++slots_[held_read_].issued;
    ++pending_;
    peak_pending_ = std::max(peak_pending_, pending_);
    held_read_ = slots_.size();
  }

  void complete(uint32_t tag, const uint8_t* data, uint32_t bytes) {
    const uint32_t index = tag / max_words_, word = tag % max_words_;
    if (index >= slots_.size()) {
      throw std::logic_error("S2G response slot out of bounds");
    }
    auto& slot = slots_[index];
    if (!slot.live || word >= slot.issued || slot.seen.at(word) || !pending_ || bytes < word_bytes_) {
      throw std::logic_error("S2G response has no matching outstanding word");
    }
    const uint64_t address = (slot.line.source & ~uint64_t(word_bytes_ - 1)) + uint64_t(word) * word_bytes_;
    const uint64_t begin = std::max(address, slot.line.source);
    const uint64_t end = std::min(address + word_bytes_, slot.line.source + slot.line.length);
    std::memcpy(slot.data.data() + slot.line.offset + begin - slot.line.source, data + begin - address, end - begin);
    slot.seen[word] = true;
    ++slot.received;
    --pending_;
  }

  bool peek_store(Line* line, const std::vector<uint8_t>** data) {
    if (held_store_ == slots_.size()) {
      for (uint32_t i = 0; i < slots_.size(); ++i) {
        if (slots_[i].live && slots_[i].received == slots_[i].words) {
          held_store_ = i;
          break;
        }
      }
    }
    if (held_store_ == slots_.size()) {
      return false;
    }
    *line = slots_[held_store_].line;
    *data = &slots_[held_store_].data;
    return true;
  }

  void accept_store() {
    if (held_store_ == slots_.size()) {
      throw std::logic_error("S2G store accepted without valid payload");
    }
    slots_[held_store_].live = false;
    held_store_ = slots_.size();
  }

  bool source_done() const {
    return last_seen_ && std::none_of(slots_.begin(), slots_.end(), [](const Slot& s) {
      return s.live && s.received != s.words;
    });
  }

  bool empty() const {
    return std::none_of(slots_.begin(), slots_.end(), [](const Slot& s) { return s.live; });
  }

  uint32_t tag_count() const { return slots_.size() * max_words_; }
  uint32_t peak_pending() const { return peak_pending_; }
  uint64_t response_address(uint32_t tag) const {
    return (slots_.at(tag / max_words_).line.source & ~uint64_t(word_bytes_ - 1))
         + uint64_t(tag % max_words_) * word_bytes_;
  }

private:
  struct Slot {
    bool live = false;
    Line line{};
    uint32_t words = 0;
    uint32_t issued = 0;
    uint32_t received = 0;
    std::vector<uint8_t> seen;
    std::vector<uint8_t> data;
  };

  std::vector<Slot> slots_;
  uint32_t credits_, word_bytes_, line_bytes_, max_words_;
  uint32_t pending_ = 0, peak_pending_ = 0;
  uint32_t held_read_, held_store_;
  bool last_seen_ = false;
};

} // namespace vortex
