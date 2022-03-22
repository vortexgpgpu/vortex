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