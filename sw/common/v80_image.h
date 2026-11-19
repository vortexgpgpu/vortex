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

#include <cerrno>
#include <cinttypes>
#include <cstdint>
#include <cstdio>
#include <fcntl.h>
#include <string>
#include <sys/file.h>
#include <sys/stat.h>
#include <unistd.h>

namespace vortex {

// The image a V80 holds, as the process that wrote it recorded it.
//
// SLASH has no query for the design in the fabric, so a driver that wants to
// skip a design write, as XRT's load_xclbin() does for an xclbin that is
// already loaded, has to keep that fact itself. The record is only ever a
// claim: a holder checks it against the device before relying on it.
struct v80_image_t {
  // Number of identity words a driver reads back from the device.
  static constexpr int kIdentWords = 6;

  uint64_t digest = 0;        // v80_image_digest() of the vbin
  uint64_t pci_instance = 0;  // v80_image_record::pci_instance() when written
  uint32_t ident[kIdentWords] = {};
};

// All-ones in every word is a bus that returned no data, not an identity.
inline bool v80_image_answers(const uint32_t words[v80_image_t::kIdentWords]) {
  for (int i = 0; i < v80_image_t::kIdentWords; ++i) {
    if (words[i] != 0xFFFFFFFFu) {
      return true;
    }
  }
  return false;
}

// What the device says about a record's claim. `words` is what it answered
// for the identity registers, null when a read failed.
enum class v80_image_check_t {
  same,       // the recorded image: no design write is needed
  different,  // another design answers: the image has to be written
  silent,     // nothing answers: not a state to write a design over
};

inline v80_image_check_t v80_image_check(const v80_image_t& recorded,
                                         const uint32_t* words) {
  if (words == nullptr || !v80_image_answers(words)) {
    return v80_image_check_t::silent;
  }
  for (int i = 0; i < v80_image_t::kIdentWords; ++i) {
    if (words[i] != recorded.ident[i]) {
      return v80_image_check_t::different;
    }
  }
  return v80_image_check_t::same;
}

// Content digest of a vbin: 64-bit FNV-1a over the file.
// Never zero for a readable file; zero when it cannot be read.
inline uint64_t v80_image_digest(const std::string& path) {
  const int fd = ::open(path.c_str(), O_RDONLY | O_CLOEXEC);
  if (fd < 0) {
    return 0;
  }
  uint64_t h = 0xcbf29ce484222325ull;
  auto mix = [&h](uint8_t b) { h = (h ^ b) * 0x100000001b3ull; };
  static constexpr size_t kChunk = 1 << 16;
  std::string buf(kChunk, '\0');
  for (;;) {
    const ssize_t n = ::read(fd, &buf[0], kChunk);
    if (n < 0) {
      if (errno == EINTR) {
        continue;
      }
      ::close(fd);
      return 0;
    }
    if (n == 0) {
      break;
    }
    for (ssize_t i = 0; i < n; ++i) {
      mix(static_cast<uint8_t>(buf[i]));
    }
  }
  ::close(fd);
  return h ? h : 1;
}

// "01:00", "0000:01:00" and either with a ".N" function all name the card
// "0000:01:00".
inline std::string v80_normalize_bdf(const std::string& bdf) {
  std::string s = bdf;
  const size_t colon = s.rfind(':');
  const size_t dot = s.rfind('.');
  if (dot != std::string::npos && colon != std::string::npos && dot > colon) {
    s.erase(dot);
  }
  if (s.find(':') == s.rfind(':')) {
    s = "0000:" + s;
  }
  return s;
}

class v80_image_record {
public:
  // The record lives in a tmpfs so that a host boot, which is also when the
  // card reloads from flash and loses its design, discards it.
  explicit v80_image_record(const std::string& bdf,
                            const std::string& dir = "/dev/shm",
                            const std::string& sysfs = "/sys/bus/pci/devices")
    : card_(v80_normalize_bdf(bdf))
    , path_(dir + "/vortex-v80-" + card_ + ".image")
    , sysfs_(sysfs)
  {}

  const std::string& path() const {
    return path_;
  }

  // Identifies this enumeration of the card, from the sysfs nodes of two of
  // its PCI functions; a function that is removed and added back gets a new
  // node. PF0 is re-added by a card reset or a hot-plug, across which the
  // fabric did not keep its design. PF2 is re-added by vrtd after every design
  // write, whichever client asked for it. Zero when either is not on the bus.
  uint64_t pci_instance() const {
    struct stat pf0, pf2;
    if (::stat((sysfs_ + "/" + card_ + ".0").c_str(), &pf0) != 0
     || ::stat((sysfs_ + "/" + card_ + ".2").c_str(), &pf2) != 0) {
      return 0;
    }
    const uint64_t instance = (static_cast<uint64_t>(pf0.st_ino) << 32)
                            ^ static_cast<uint64_t>(pf2.st_ino);
    return instance ? instance : 1;
  }

  // True when the record says this enumeration of the card holds the image
  // with this digest; *image then carries the identity recorded with it.
  bool holds(uint64_t digest, v80_image_t* image) const {
    const uint64_t instance = this->pci_instance();
    if (instance == 0) {
      return false;
    }
    v80_image_t rec;
    if (!this->load(&rec) || rec.digest != digest || rec.pci_instance != instance) {
      return false;
    }
    *image = rec;
    return true;
  }

  bool load(v80_image_t* image) const {
    const int fd = open_regular(O_RDONLY);
    if (fd < 0) {
      return false;
    }
    char text[256] = {};
    ssize_t n = -1;
    if (::flock(fd, LOCK_SH) == 0) {
      n = ::read(fd, text, sizeof(text) - 1);
    }
    ::close(fd);
    if (n <= 0) {
      return false;
    }
    v80_image_t rec;
    int end = 0;
    const int fields = sscanf(text,
        "vortex-v80-image 1 %" SCNx64 " %" SCNx64
        " %" SCNx32 " %" SCNx32 " %" SCNx32 " %" SCNx32 " %" SCNx32 " %" SCNx32 " %n",
        &rec.digest, &rec.pci_instance,
        &rec.ident[0], &rec.ident[1], &rec.ident[2],
        &rec.ident[3], &rec.ident[4], &rec.ident[5], &end);
    if (fields != 2 + v80_image_t::kIdentWords || end != n || rec.digest == 0) {
      return false;
    }
    *image = rec;
    return true;
  }

  bool store(const v80_image_t& image) const {
    char text[256];
    const int n = snprintf(text, sizeof(text),
        "vortex-v80-image 1 %" PRIx64 " %" PRIx64
        " %" PRIx32 " %" PRIx32 " %" PRIx32 " %" PRIx32 " %" PRIx32 " %" PRIx32 "\n",
        image.digest, image.pci_instance,
        image.ident[0], image.ident[1], image.ident[2],
        image.ident[3], image.ident[4], image.ident[5]);
    const int fd = this->open_for_write(true);
    if (fd < 0) {
      return false;
    }
    const bool ok = ::flock(fd, LOCK_EX) == 0
                 && ::ftruncate(fd, 0) == 0
                 && ::write(fd, text, n) == n;
    if (!ok) {
      // A partial record must not survive as a claim.
      (void)!::ftruncate(fd, 0);
    }
    ::close(fd);
    return ok;
  }

  // Voids the record. True when no claim is left, which includes there being
  // no record at all.
  bool clear() const {
    const int fd = this->open_for_write(false);
    if (fd < 0) {
      return errno == ENOENT;
    }
    const bool ok = ::flock(fd, LOCK_EX) == 0 && ::ftruncate(fd, 0) == 0;
    ::close(fd);
    return ok;
  }

private:
  // Only a regular file is a record. Nothing else at the path is followed,
  // waited on or written to.
  int open_regular(int flags, mode_t mode = 0) const {
    const int fd = ::open(path_.c_str(), flags | O_NOFOLLOW | O_NONBLOCK | O_CLOEXEC, mode);
    if (fd < 0) {
      return -1;
    }
    struct stat st;
    if (::fstat(fd, &st) != 0 || !S_ISREG(st.st_mode)) {
      ::close(fd);
      errno = EINVAL;
      return -1;
    }
    return fd;
  }

  // The record is rewritten in place and is writable by every user: whoever
  // programs the card must be able to void a claim somebody else left, and a
  // sticky directory lets only the owner replace or unlink the file. An
  // existing record is opened without O_CREAT, which such a directory refuses
  // for a file of another owner.
  int open_for_write(bool create) const {
    for (int attempt = 0; attempt < 2; ++attempt) {
      int fd = open_regular(O_WRONLY);
      if (fd >= 0 || errno != ENOENT || !create) {
        return fd;
      }
      fd = open_regular(O_WRONLY | O_CREAT | O_EXCL, 0666);
      if (fd >= 0) {
        (void)!::fchmod(fd, 0666);   // the umask must not narrow this file
        return fd;
      }
      if (errno != EEXIST) {
        return -1;
      }
    }
    return -1;
  }

  std::string card_;
  std::string path_;
  std::string sysfs_;
};

} // namespace vortex
