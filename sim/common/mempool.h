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

#include <stack>

template <typename T>
class MemoryPool {
public:  
  MemoryPool(uint32_t max_size) : max_size_(max_size) {}

  MemoryPool(MemoryPool && other) 
    : free_list_(std::move(other.free_list_)) 
  {}

  ~MemoryPool() {
    this->flush();
  }

  void* allocate() {
    void* mem;
    if (!free_list_.empty()) {
      auto entry = free_list_.top();
      free_list_.pop();
      mem = static_cast<void*>(entry);      
    } else {
      mem = ::operator new(sizeof(T));
    }
    return mem;
  }

  void deallocate(void * object) {
    if (free_list_.size() < max_size_) {
      free_list_.push(static_cast<T*>(object));
    } else {
      ::operator delete(object);
    }
  }

  void flush() {
    while (!free_list_.empty()) {
      auto entry = free_list_.top();
      free_list_.pop();
      ::operator delete(entry);      
    }
  }

private:
  std::stack<T*> free_list_;
  uint32_t max_size_;
};
