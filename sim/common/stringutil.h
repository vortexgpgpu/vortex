#pragma once

#include <iostream>
#include  <iomanip>

class ByteStream : public std::istream {
private:
  const void *buf_;
  std::size_t size_;

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
};

class IndentStream : public std::streambuf {
private:
  std::streambuf* dest_;
  bool            isBeginLine_;
  std::string     indent_;
  std::ostream*   owner_;

protected:
  virtual int overflow(int ch) {
    if (isBeginLine_ && ch != '\n') {
      dest_->sputn(indent_.data(), indent_.size());
    }
    isBeginLine_ = ch == '\n';
    return dest_->sputc(ch);
  }

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
};