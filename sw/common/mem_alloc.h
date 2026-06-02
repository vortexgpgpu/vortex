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

#include <cstdint>
#include <assert.h>
#include <stdio.h>

namespace vortex {

class MemoryAllocator {
public:
  MemoryAllocator(
    uint64_t baseAddress,
    uint64_t capacity,
    uint32_t pageAlign,
    uint32_t blockAlign)
    : baseAddress_(baseAddress)
    , capacity_(capacity)
    , pageAlign_(pageAlign)
    , blockAlign_(blockAlign)
    , pages_(nullptr)
    , allocated_(0)
  {}

  ~MemoryAllocator() {
    page_t* currPage = pages_;
    while (currPage) {
      auto nextPage = currPage->next;
      #ifdef VX_CFG_VM_ENABLE
      block_t* currblock = currPage->findfirstUsedBlock();
      block_t* nextblock;
      while (currblock) {
        nextblock= currblock->nextUsed;
        currPage->release(currblock);
        currblock = nextblock;
      }
      #endif
      delete currPage;
      currPage = nextPage;
    }
  }

  uint32_t baseAddress() const {
    return baseAddress_;
  }

  uint32_t capacity() const {
    return capacity_;
  }

  uint64_t free() const {
    return (capacity_ - allocated_);
  }

  uint64_t allocated() const {
    return allocated_;
  }

  int reserve(uint64_t addr, uint64_t size) {
    if (size == 0) {
      printf("Error: invalid arguments\n");
      return -1;
    }

    size = alignSize(size, pageAlign_);

    // Check if the reservation is within memory capacity bounds
    if (addr + size > baseAddress_ + capacity_) {
      printf("Error: address range out of bounds - requested=0x%lx, base+capacity=0x%lx\n", (addr + size), (baseAddress_ +capacity_));
      return -1;
    }

    // Ensure the reservation does not overlap with existing pages
    uint64_t overlapStart, overlapEnd;
    if (hasPageOverlap(addr, size, &overlapStart, &overlapEnd)) {
      printf("Error: address range overlaps with existing allocation - requested=[0x%lx-0x%lx], existing=[0x%lx, 0x%lx]\n", addr, addr+size, overlapStart, overlapEnd);
      return -1;
    }

    auto newPage = this->createPage(addr, size);
    auto freeBlock = newPage->findFreeBlock(size);
    newPage->allocate(size, freeBlock);
    allocated_ += size;

    return 0;
  }

  int allocate(uint64_t size, uint64_t* addr) {
    if (size == 0 || addr == nullptr) {
      printf("Error: invalid arguments\n");
      return -1;
    }

    size = alignSize(size, blockAlign_);

    // Walk pages to find a free block
    block_t* freeBlock = nullptr;
    auto currPage = pages_;
    while (currPage) {
      freeBlock = currPage->findFreeBlock(size);
      if (freeBlock != nullptr)
        break;
      currPage = currPage->next;
    }

    // Allocate a new page if no free block is found
    if (freeBlock == nullptr) {
      auto pageSize = alignSize(size, pageAlign_);
      uint64_t pageAddr;
      if (!this->findNextAddress(pageSize, &pageAddr)) {
        printf("Error: out of memory (Can't find next address)\n");
        return -1;
      }
      currPage = this->createPage(pageAddr, pageSize);
      if (nullptr == currPage) {
        printf("Error: out of memory (Can't create a page)\n");
        return -1;
      }
      freeBlock = currPage->findFreeBlock(size);
    }

    currPage->allocate(size, freeBlock);
    *addr = freeBlock->addr;
    allocated_ += size;

    return 0;
  }

  int release(uint64_t addr) {
    block_t* usedBlock = nullptr;
    auto currPage = pages_;
    while (currPage) {
      usedBlock = currPage->findUsedBlock(addr);
      if (usedBlock != nullptr)
        break;
      currPage = currPage->next;
    }

    if (nullptr == usedBlock) {
      printf("warning: release address not found: 0x%lx\n", addr);
      return -1;
    }

    auto size = usedBlock->size;
    currPage->release(usedBlock);
    if (currPage->empty()) {
      this->deletePage(currPage);
    }
    allocated_ -= size;

    return 0;
  }

private:

  struct block_t {
    block_t* nextFreeS;
    block_t* prevFreeS;

    block_t* nextFreeM;
    block_t* prevFreeM;

    block_t* nextUsed;
    block_t* prevUsed;

    uint64_t addr;
    uint64_t size;

    block_t(uint64_t addr, uint64_t size)
      : nextFreeS(nullptr)
      , prevFreeS(nullptr)
      , nextFreeM(nullptr)
      , prevFreeM(nullptr)
      , nextUsed(nullptr)
      , prevUsed(nullptr)
      , addr(addr)
      , size(size)
    {}
  };

  struct page_t {
    page_t*  next;
    uint64_t addr;
    uint64_t size;

    page_t(uint64_t addr, uint64_t size, uint32_t blockAlign) :
      next(nullptr),
      addr(addr),
      size(size),
      blockAlign_(blockAlign),
      usedList_(nullptr) {
      freeSList_ = freeMList_ = new block_t(addr, size);
    }

    ~page_t() {
      // The page should be empty
      assert(nullptr == usedList_);
      assert(freeMList_
          && (nullptr == freeMList_->nextFreeM)
          && (nullptr == freeMList_->prevFreeM));
      delete freeMList_;
    }

    bool empty() const {
      return (usedList_ == nullptr);
    }

    void allocate(uint64_t size, block_t* freeBlock) {
      this->removeFreeMList(freeBlock);
      this->removeFreeSList(freeBlock);

      // Split the free block if larger than needed.
      uint64_t extraBytes = freeBlock->size - size;
      if (extraBytes >= blockAlign_) {
        freeBlock->size = size;
        auto nextAddr = freeBlock->addr + size;
        auto newBlock = new block_t(nextAddr, extraBytes);
        this->insertFreeMList(newBlock);
        this->insertFreeSList(newBlock);
      }

      this->insertUsedList(freeBlock);
    }

    void release(block_t* usedBlock) {
      this->removeUsedList(usedBlock);
      this->insertFreeMList(usedBlock);

      // Merge adjacent free blocks from the left.
      if (usedBlock->prevFreeM) {
        auto prevAddr = usedBlock->prevFreeM->addr + usedBlock->prevFreeM->size;
        if (usedBlock->addr == prevAddr) {
          auto prevBlock = usedBlock->prevFreeM;
          prevBlock->size += usedBlock->size;
          prevBlock->nextFreeM = usedBlock->nextFreeM;
          if (prevBlock->nextFreeM) {
            prevBlock->nextFreeM->prevFreeM = prevBlock;
          }
          // Detach prev from S-list since its size grew.
          this->removeFreeSList(prevBlock);
          delete usedBlock;
          usedBlock = prevBlock;
        }
      }

      // Merge adjacent free blocks from the right.
      if (usedBlock->nextFreeM) {
        auto nextAddr = usedBlock->addr + usedBlock->size;
        if (usedBlock->nextFreeM->addr == nextAddr) {
          auto nextBlock = usedBlock->nextFreeM;
          usedBlock->size += nextBlock->size;
          usedBlock->nextFreeM = nextBlock->nextFreeM;
          if (usedBlock->nextFreeM) {
            usedBlock->nextFreeM->prevFreeM = usedBlock;
          }
          this->removeFreeSList(nextBlock);
          delete nextBlock;
        }
      }

      this->insertFreeSList(usedBlock);
    }

    block_t* findFreeBlock(uint64_t size) {
      auto freeBlock = freeSList_;
      if (freeBlock) {
        // S-list is sorted largest-first; find the smallest block that fits.
        if (freeBlock->size >= size) {
          while (freeBlock->nextFreeS && (freeBlock->nextFreeS->size >= size)) {
            freeBlock = freeBlock->nextFreeS;
          }
          return freeBlock;
        }
      }
      return nullptr;
    }

    block_t* findUsedBlock(uint64_t addr) {
      if (addr >= this->addr && addr < (this->addr + this->size)) {
        auto useBlock = usedList_;
        while (useBlock) {
          if (useBlock->addr == addr)
            return useBlock;
          useBlock = useBlock->nextUsed;
        }
      }
      return nullptr;
    }
#ifdef VX_CFG_VM_ENABLE
    block_t* findfirstUsedBlock() {
      return usedList_;
    }
#endif

  private:

    void insertUsedList(block_t* block) {
      block->nextUsed = usedList_;
      if (usedList_) {
        usedList_->prevUsed = block;
      }
      usedList_ = block;
    }

    void removeUsedList(block_t* block) {
      if (block->prevUsed) {
        block->prevUsed->nextUsed = block->nextUsed;
      } else {
        usedList_ = block->nextUsed;
      }
      if (block->nextUsed) {
        block->nextUsed->prevUsed = block->prevUsed;
      }
      block->nextUsed = nullptr;
      block->prevUsed = nullptr;
    }

    void insertFreeMList(block_t* block) {
      block_t* currBlock = freeMList_;
      block_t* prevBlock = nullptr;
      while (currBlock && (currBlock->addr < block->addr)) {
        prevBlock = currBlock;
        currBlock = currBlock->nextFreeM;
      }
      block->nextFreeM = currBlock;
      block->prevFreeM = prevBlock;
      if (prevBlock) {
        prevBlock->nextFreeM = block;
      } else {
        freeMList_ = block;
      }
      if (currBlock) {
        currBlock->prevFreeM = block;
      }
    }

    void removeFreeMList(block_t* block) {
      if (block->prevFreeM) {
        block->prevFreeM->nextFreeM = block->nextFreeM;
      } else {
        freeMList_ = block->nextFreeM;
      }
      if (block->nextFreeM) {
        block->nextFreeM->prevFreeM = block->prevFreeM;
      }
      block->nextFreeM = nullptr;
      block->prevFreeM = nullptr;
    }

    void insertFreeSList(block_t* block) {
      block_t* currBlock = freeSList_;
      block_t* prevBlock = nullptr;
      while (currBlock && (currBlock->size > block->size)) {
        prevBlock = currBlock;
        currBlock = currBlock->nextFreeS;
      }
      block->nextFreeS = currBlock;
      block->prevFreeS = prevBlock;
      if (prevBlock) {
        prevBlock->nextFreeS = block;
      } else {
        freeSList_ = block;
      }
      if (currBlock) {
        currBlock->prevFreeS = block;
      }
    }

    void removeFreeSList(block_t* block) {
      if (block->prevFreeS) {
        block->prevFreeS->nextFreeS = block->nextFreeS;
      } else {
        freeSList_ = block->nextFreeS;
      }
      if (block->nextFreeS) {
        block->nextFreeS->prevFreeS = block->prevFreeS;
      }
      block->nextFreeS = nullptr;
      block->prevFreeS = nullptr;
    }

    uint32_t blockAlign_;
    block_t* usedList_;
    block_t* freeSList_; // sorted by decreasing size (for allocation)
    block_t* freeMList_; // sorted by increasing address (for merging)
  };

  page_t* createPage(uint64_t addr, uint64_t size) {
    auto newPage = new page_t(addr, size, blockAlign_);

    // Insert in address-sorted order.
    if (pages_ == nullptr || pages_->addr > newPage->addr) {
      newPage->next = pages_;
      pages_ = newPage;
    } else {
      page_t* current = pages_;
      while (current->next != nullptr && current->next->addr < newPage->addr) {
        current = current->next;
      }
      newPage->next = current->next;
      current->next = newPage;
    }

    return newPage;
  }

  void deletePage(page_t* page) {
    page_t* prevPage = nullptr;
    auto currPage = pages_;
    while (currPage) {
      if (currPage == page) {
        if (prevPage) {
          prevPage->next = currPage->next;
        } else {
          pages_ = currPage->next;
        }
        break;
      }
      prevPage = currPage;
      currPage = currPage->next;
    }
    delete page;
  }

  bool findNextAddress(uint64_t size, uint64_t* addr) {
    if (pages_ == nullptr) {
      *addr = baseAddress_;
      return true;
    }

    page_t* current = pages_;
    uint64_t endOfLastPage = baseAddress_;

    while (current != nullptr) {
      uint64_t startOfCurrentPage = current->addr;
      if ((endOfLastPage + size) <= startOfCurrentPage) {
        *addr = endOfLastPage;
        return true;
      }
      endOfLastPage = current->addr + current->size;
      current = current->next;
    }

    if ((endOfLastPage + size) <= (baseAddress_ + capacity_)) {
      *addr = endOfLastPage;
      return true;
    }

    return false;
  }

  bool hasPageOverlap(uint64_t start, uint64_t size, uint64_t* overlapStart, uint64_t* overlapEnd) {
    page_t* current = pages_;
    while (current != nullptr) {
      uint64_t pageStart = current->addr;
      uint64_t pageEnd = pageStart + current->size;
      uint64_t end = start + size;
      // Half-open range overlap: [start, end) vs [pageStart, pageEnd).
      // Adjacent ranges (start == pageEnd or end == pageStart) do NOT
      // overlap.
      if ((start < pageEnd) && (end > pageStart)) {
        *overlapStart = pageStart;
        *overlapEnd = pageEnd;
        return true;
      }
      current = current->next;
    }
    return false;
  }

  static uint64_t alignSize(uint64_t size, uint64_t alignment) {
    assert(0 == (alignment & (alignment - 1)));
    return (size + alignment - 1) & ~(alignment - 1);
  }

  uint64_t baseAddress_;
  uint64_t capacity_;
  uint32_t pageAlign_;
  uint32_t blockAlign_;
  page_t*  pages_;
  uint64_t nextAddress_;
  uint64_t allocated_;
};

} // namespace vortex
