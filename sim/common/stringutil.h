// Copyright © 2019-2023
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#pragma once

#include <iostream>
#include  <iomanip>

class ByteStream : public std::istream {
public:
  ByteStream(const void *buf, std::size_t size) : buf_(buf), size_(size) {}

  friend std::ostream& operator<<(std::ostream& os, const ByteStream& obj) {
    auto oldflags = os.flags();
    auto oldwidth = os.width();
    auto oldfill  = os.fill();
    for (std::size_t i = 0, n = obj.size_; i < n; ++i) {
      int byte = *((uint8_t*)obj.buf_ + (n - 1 - i));
      os << std::hex << std::setw(2) << std::setfill('0') << byte;
    }
    os.fill(oldfill);
    os.width(oldwidth);
    os.flags(oldflags);
    return os;
  }

private:
  const void *buf_;
  std::size_t size_;
};

class IndentStream : public std::streambuf {
public:
  explicit IndentStream(std::streambuf* dest, int indent = 4)
    : dest_(dest)
    , isBeginLine_(true)
    , indent_(indent, ' ')
    , owner_(nullptr)
  {}

  explicit IndentStream(std::ostream& dest, int indent = 4)
    : dest_(dest.rdbuf())
    , isBeginLine_(true)
    , indent_(indent, ' ')
    , owner_(&dest) {
      owner_->rdbuf(this);
  }

  virtual ~IndentStream() {
    if (owner_)
        owner_->rdbuf(dest_);
  }

protected:
  virtual int overflow(int ch) {
    if (isBeginLine_ && ch != '\n') {
      dest_->sputn(indent_.data(), indent_.size());
    }
    isBeginLine_ = ch == '\n';
    return dest_->sputc(ch);
  }

private:
  std::streambuf* dest_;
  bool            isBeginLine_;
  std::string     indent_;
  std::ostream*   owner_;
};

template <typename... Args>
std::string StrFormat(const std::string& fmt, Args... args) {
  auto size = std::snprintf(nullptr, 0, fmt.c_str(), args...) + 1;
  if (size <= 0) {
    throw std::runtime_error("Error during formatting.");
  }
  std::vector<char> buf(size);
  std::snprintf(buf.data(), size, fmt.c_str(), args...);
  return std::string(buf.data(), buf.data() + size - 1);
}