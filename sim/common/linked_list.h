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

#include <assert.h>
#include <cstdint>

namespace vortex {

template <typename T>
struct LinkedListNode {
  LinkedListNode* next{nullptr};
  LinkedListNode* prev{nullptr};
  T* object{nullptr};
};

template <typename T, LinkedListNode<T> T::*HookPtr>
class LinkedList {
public:

  bool empty() const {
    return head_ == nullptr;
  }

  size_t size() const {
    return size_;
  }

  class iterator {
  public:
    explicit iterator(LinkedListNode<T> *node) : current_(node) {}

    T &operator*() const { return *current_->object; }
    T *operator->() const { return current_->object; }

    iterator &operator++() {
      current_ = current_->next;
      return *this;
    }

    iterator operator++(int) {
      iterator tmp = *this;
      ++(*this);
      return tmp;
    }

    bool operator!=(const iterator &other) const {
      return current_ != other.current_;
    }

  private:
    LinkedListNode<T>* current_;
    friend class LinkedList;
  };

  class const_iterator {
  public:
    explicit const_iterator(const LinkedListNode<T> *node)
    : current_(const_cast<LinkedListNode<T>*>(node)) {}

    const_iterator(const iterator &other) : current_(other.current_) {}

    const T &operator*() const { return *current_->object; }
    const T *operator->() const { return current_->object; }

    const_iterator &operator++() {
      current_ = current_->next;
      return *this;
    }

    const_iterator operator++(int) {
      const_iterator tmp = *this;
      ++(*this);
      return tmp;
    }

    bool operator!=(const const_iterator &other) const {
      return current_ != other.current_;
    }

  private:
    LinkedListNode<T>* current_;
    friend class LinkedList;
  };

  class reverse_iterator {
  public:
    explicit reverse_iterator(LinkedListNode<T>* node) : current_(node) {}

    T& operator*() const { return *current_->object; }
    T* operator->() const { return current_->object; }

    reverse_iterator& operator++() {
      current_ = current_->prev;
      return *this;
    }

    reverse_iterator operator++(int) {
      reverse_iterator tmp = *this;
      ++(*this);
      return tmp;
    }

    bool operator!=(const reverse_iterator& other) const {
      return current_ != other.current_;
    }

  private:
    LinkedListNode<T>* current_;
    friend class LinkedList;
  };

  class const_reverse_iterator {
  public:
    explicit const_reverse_iterator(const LinkedListNode<T>* node)
    : current_(const_cast<LinkedListNode<T>*>(node)) {}

    const_reverse_iterator(const reverse_iterator& other) : current_(other.current_) {}

    const T& operator*() const { return *current_->object; }
    const T* operator->() const { return current_->object; }

    const_reverse_iterator& operator++() {
      current_ = current_->prev;
      return *this;
    }

    const_reverse_iterator operator++(int) {
      const_reverse_iterator tmp = *this;
      ++(*this);
      return tmp;
    }

    bool operator!=(const const_reverse_iterator& other) const {
      return current_ != other.current_;
    }

  private:
    LinkedListNode<T>* current_;
    friend class LinkedList;
  };

  iterator begin() const { return iterator(head_); }
  iterator end() const { return iterator(nullptr); }

  const_iterator cbegin() { return const_iterator(head_); }
  const_iterator cend() { return const_iterator(nullptr); }

  reverse_iterator rbegin() { return reverse_iterator(tail_); }
  reverse_iterator rend() { return reverse_iterator(nullptr); }

  const_reverse_iterator crbegin() { return const_reverse_iterator(tail_); }
  const_reverse_iterator crend() { return const_reverse_iterator(nullptr); }

  iterator insert(iterator pos, T *obj) {
    assert(obj != nullptr);
    auto node = &(obj->*HookPtr);
    node->object = obj;
    if (pos.current_ == nullptr) {
      if (tail_ == nullptr) {
        // If the list is empty, set both head and tail to the new node
        head_ = tail_ = node;
        node->prev = nullptr;
        node->next = nullptr;
      } else {
        // Insert at the end, after tail
        node->prev = tail_;
        node->next = nullptr;
        tail_->next = node;
        tail_ = node;
      }
    } else if (pos.current_ == head_) {
      // If inserting at the head
      node->next = head_;
      node->prev = nullptr;
      head_->prev = node;
      head_ = node;
    } else {
      // Inserting in the middle
      node->prev = pos.current_->prev;
      node->next = pos.current_;
      pos.current_->prev->next = node;
      pos.current_->prev = node;
    }
    ++size_;
    return iterator(node);
  }

  iterator erase(iterator pos) {
    assert(pos.current_ != nullptr);
    auto node = pos.current_;
    iterator it_next(node->next);
    if (node->prev) {
      node->prev->next = node->next;
    } else {
      head_ = node->next;
    }
    if (node->next) {
      node->next->prev = node->prev;
    } else {
      tail_ = node->prev;
    }
    node->prev = node->next = nullptr;
    node->object = nullptr;
    --size_;
    return it_next;
  }

  T* front() {
    assert(head_ != nullptr);
    return head_->object;
  }

  const T* front() const {
    assert(head_ != nullptr);
    return head_->object;
  }

  T* back() {
    assert(tail_ != nullptr);
    return tail_->object;
  }

  const T* back() const {
    assert(tail_ != nullptr);
    return tail_->object;
  }

  void push_back(T *obj) {
    insert(iterator(tail_), obj);
  }

  void push_front(T *obj) {
    insert(iterator(head_), obj);
  }

  void pop_front() {
    erase(iterator(head_));
  }

  void pop_back() {
    erase(iterator(tail_));
  }

  void remove(T *obj) {
    auto node = &(obj->*HookPtr);
    erase(iterator(node));
  }

  size_t count(const T* obj) const {
    assert(obj != nullptr);
    const auto* node = &(obj->*HookPtr);
    return (node->object == obj) &&
           (node->next != nullptr || node->prev != nullptr || head_ == node) ? 1 : 0;
  }

  const_iterator find(const T* obj) {
    assert(obj != nullptr);
    auto* node = &(obj->*HookPtr);
    if (node->object != obj)
      return this->end();
    return const_iterator(node);
  }

  iterator find(T* obj) {
    assert(obj != nullptr);
    auto* node = &(obj->*HookPtr);
    if (node->object != obj)
      return this->end();
    return iterator(node);
  }

  void clear() {
    while (head_) {
      auto node = head_;
      head_ = head_->next;
      node->object = nullptr;
      node->next = node->prev = nullptr;
    }
    tail_ = nullptr;
    size_ = 0;
  }

private:

  LinkedListNode<T>* head_{nullptr};
  LinkedListNode<T>* tail_{nullptr};
  size_t size_{0};
};

} // namespace vortex