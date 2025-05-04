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

#include "util.h"
#include <fstream>
#include <sstream>
#include <string.h>

// return file extension
const char* fileExtension(const char* filepath) {
  const char *ext = strrchr(filepath, '.');
  if (ext == NULL || ext == filepath)
    return "";
  return ext + 1;
}

void* aligned_malloc(size_t size, size_t alignment) {
  // reserve margin for alignment and storing of unaligned address
  assert((alignment & (alignment - 1)) == 0);   // Power of 2 alignment.
  size_t margin = (alignment-1) + sizeof(void*);
  void *unaligned_addr = malloc(size + margin);
  void **aligned_addr = (void**)((uintptr_t)(((uint8_t*)unaligned_addr) + margin) & ~(alignment-1));
  aligned_addr[-1] = unaligned_addr;
  return aligned_addr;
}

void aligned_free(void *ptr) {
  // retreive the stored unaligned address and use it to free the allocation
  void* unaligned_addr = ((void**)ptr)[-1];
  free(unaligned_addr);
}

std::string vortex::resolve_file_path(const std::string& filename, const std::string& searchPaths) {
  std::ifstream ifs(filename);
  if (!ifs) {
    std::stringstream ss(searchPaths);
    std::string path;
    while (std::getline(ss, path, ',')) {
      if (!path.empty()) {
        std::string filePath = path + "/" + filename;
        std::ifstream ifs(filePath);
        if (ifs)
          return filePath;
      }
    }
  }
  return filename;
}