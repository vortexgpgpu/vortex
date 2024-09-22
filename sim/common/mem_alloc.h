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
    // Free allocated pages
    page_t* currPage = pages_;
    while (currPage) {
      auto nextPage = currPage->next;
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
      printf("error: invalid arguments\n");
      return -1;
    }

    // Align allocation size
    size = alignSize(size, pageAlign_);

    // Check if the reservation is within memory capacity bounds
    if (addr + size > capacity_) {
      printf("error: address range out of bounds - requested=0x%lx, capacity=0x%lx\n", (addr + size), capacity_);
      return -1;
    }

    // Ensure the reservation does not overlap with existing pages
    uint64_t overlapStart, overlapEnd;
    if (hasPageOverlap(addr, size, &overlapStart, &overlapEnd)) {
      printf("error: address range overlaps with existing allocation - requested=[0x%lx-0x%lx], existing=[0x%lx, 0x%lx]\n", addr, addr+size, overlapStart, overlapEnd);
      return -1;
    }

    // allocate a new page for segment
    auto newPage = this->createPage(addr, size);

    // allocate space on free block
    auto freeBlock = newPage->findFreeBlock(size);
    newPage->allocate(size, freeBlock);

    // Update allocated size
    allocated_ += size;

    return 0;
  }

  int allocate(uint64_t size, uint64_t* addr) {
    if (size == 0 || addr == nullptr) {
      printf("error: invalid arguments\n");
      return -1;
    }

    // Align allocation size
    size = alignSize(size, blockAlign_);

    // Walk thru all pages to find a free block
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
        printf("error: out of memory\n");
        return -1;
      }
      currPage = this->createPage(pageAddr, pageSize);
      if (nullptr == currPage) {
        printf("error: out of memory\n");
        return -1;
      }
      freeBlock = currPage->findFreeBlock(size);
    }

    // allocate space on free block
    currPage->allocate(size, freeBlock);

    // Return the free block address
    *addr = freeBlock->addr;

    // Update allocated size
    allocated_ += size;

    return 0;
  }

  int release(uint64_t addr) {
    // Walk all pages to find the pointer
    block_t* usedBlock = nullptr;
    auto currPage = pages_;
    while (currPage) {
      usedBlock = currPage->findUsedBlock(addr);
      if (usedBlock != nullptr)
        break;
      currPage = currPage->next;
    }

    // found the corresponding block?
    if (nullptr == usedBlock) {
      printf("warning: release address not found: 0x%lx\n", addr);
      return -1;
    }

    auto size = usedBlock->size;

    // release the used block
    currPage->release(usedBlock);

    // Free the page if empty
    if (currPage->empty()) {
      this->deletePage(currPage);
    }

    // update allocated size
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
      // Remove the block from the free lists
      this->removeFreeMList(freeBlock);
      this->removeFreeSList(freeBlock);

      // If the free block we have found is larger than what we are looking for,
      // we may be able to split our free block in two.
      uint64_t extraBytes = freeBlock->size - size;
      if (extraBytes >= blockAlign_) {
        // Reduce the free block size to the requested value
        freeBlock->size = size;

        // Allocate a new block to contain the extra buffer
        auto nextAddr = freeBlock->addr + size;
        auto newBlock = new block_t(nextAddr, extraBytes);

        // Add the new block to the free lists
        this->insertFreeMList(newBlock);
        this->insertFreeSList(newBlock);
      }

      // Insert the free block into the used list
      this->insertUsedList(freeBlock);
    }

    void release(block_t* usedBlock) {
      // Remove the block from the used list
      this->removeUsedList(usedBlock);

      // Insert the block into the free M-list.
      this->insertFreeMList(usedBlock);

      // Check if we can merge adjacent free blocks from the left.
      if (usedBlock->prevFreeM) {
        // Calculate the previous address
        auto prevAddr = usedBlock->prevFreeM->addr + usedBlock->prevFreeM->size;
        if (usedBlock->addr == prevAddr) {
          auto prevBlock = usedBlock->prevFreeM;

          // Merge the blocks to the left
          prevBlock->size += usedBlock->size;
          prevBlock->nextFreeM = usedBlock->nextFreeM;
          if (prevBlock->nextFreeM) {
            prevBlock->nextFreeM->prevFreeM = prevBlock;
          }

          // Detach previous block from the free S-list since size increased
          this->removeFreeSList(prevBlock);

          // reset usedBlock
          delete usedBlock;
          usedBlock = prevBlock;
        }
      }

      // Check if we can merge adjacent free blocks from the right.
      if (usedBlock->nextFreeM) {
        // Calculate the next allocation start address
        auto nextAddr = usedBlock->addr + usedBlock->size;
        if (usedBlock->nextFreeM->addr == nextAddr) {
          auto nextBlock = usedBlock->nextFreeM;

          // Merge the blocks to the right
          usedBlock->size += nextBlock->size;
          usedBlock->nextFreeM = nextBlock->nextFreeM;
          if (usedBlock->nextFreeM) {
            usedBlock->nextFreeM->prevFreeM = usedBlock;
          }

          // Delete next block
          this->removeFreeSList(nextBlock);
          delete nextBlock;
        }
      }

      // Insert the block into the free S-list.
      this->insertFreeSList(usedBlock);
    }

    block_t* findFreeBlock(uint64_t size) {
      auto freeBlock = freeSList_;
      if (freeBlock) {
        // The free S-list is already sorted with the largest block first
        // Quick check if the head block has enough space.
        if (freeBlock->size >= size) {
          // Find the smallest matching block in the S-list
          while (freeBlock->nextFreeS && (freeBlock->nextFreeS->size >= size)) {
            freeBlock = freeBlock->nextFreeS;
          }
          // Return the free block
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

    // block alignment
    uint32_t blockAlign_;

    // List of used blocks
    block_t* usedList_;

    // List with blocks sorted by decreasing sizes
    // Used for block lookup during memory allocation.
    block_t* freeSList_;

    // List with blocks sorted by increasing memory addresses
    // Used for block merging during memory release.
    block_t* freeMList_;
  };

  page_t* createPage(uint64_t addr, uint64_t size) {
    // Allocate object
    auto newPage = new page_t(addr, size, blockAlign_);

    // Insert the new page into the list in address sorted order
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
    // Remove the page from the list
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
    // Delete the page
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
      // Update the end of the last page to the end of the current page
      // Move to the next page in the sorted list
      endOfLastPage = current->addr + current->size;
      current = current->next;
    }

    // If no suitable gap is found, place the new page at the end of the last page
    // Check if the allocator has enough capacity
    if ((endOfLastPage + size) <= capacity_) {
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
      if ((start <= pageEnd) && (end >= pageStart)) {
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
