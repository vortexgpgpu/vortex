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

#include <dirent.h>
#include <fstream>
#include <string>

namespace vortex {

// Finds an AMD Alveo V80 the way XRT enumerates by index: scans sysfs for the
// SLASH control function (10ee:50c1) and returns its bus:device in the "01:00"
// form VRT takes. Empty when no card is on the bus.
inline std::string discover_v80_bdf() {
  DIR* d = opendir("/sys/bus/pci/devices");
  if (d == nullptr) {
    return "";
  }
  std::string found;
  while (struct dirent* e = readdir(d)) {
    const std::string name = e->d_name;              // "0000:01:00.1"
    if (name.size() < 12 || name[0] == '.') {
      continue;
    }
    auto read_hex = [&](const char* leaf) -> unsigned {
      std::ifstream f("/sys/bus/pci/devices/" + name + "/" + leaf);
      unsigned v = 0;
      f >> std::hex >> v;
      return v;
    };
    if (read_hex("vendor") == 0x10ee && read_hex("device") == 0x50c1) {
      found = name.substr(0, name.rfind('.'));       // drop the function
      if (found.rfind("0000:", 0) == 0) {
        found = found.substr(5);                     // "01:00" form
      }
      break;
    }
  }
  closedir(d);
  return found;
}

} // namespace vortex
