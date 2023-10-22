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
        , nextAddress_(0)
        , allocated_(0)
    {}

    ~MemoryAllocator() {
        // Free allocated pages
        page_t* currPage = pages_;
        while (currPage) {
            auto nextPage = currPage->next;
            this->DeletePage(currPage);
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

    int allocate(uint64_t size, uint64_t* addr) {
        if (size == 0 || addr == nullptr) {
            printf("error: invalid argurments\n");
            return -1;
        }

        // Align allocation size
        size = AlignSize(size, blockAlign_);

        // Walk thru all pages to find a free block
        block_t* freeBlock = nullptr;
        auto currPage = pages_;
        while (currPage) {
            auto currBlock = currPage->freeSList;
            if (currBlock) {
                // The free S-list is already sorted with the largest block first
                // Quick check if the head block has enough space.
                if (currBlock->size >= size) {
                    // Find the smallest matching block in the S-list
                    while (currBlock->nextFreeS 
                        && (currBlock->nextFreeS->size >= size)) {
                        currBlock = currBlock->nextFreeS;
                    }
                    // Return the free block
                    freeBlock = currBlock;
                    break;
                }
            }
            currPage = currPage->next;
        }

        if (nullptr == freeBlock) {
            // Allocate a new page for this request
            currPage = this->NewPage(size);
            if (nullptr == currPage) {
                printf("error: out of memory\n");
                return -1;
            }
            freeBlock = currPage->freeSList;
        }   

        // Remove the block from the free lists
        assert(freeBlock->size >= size);
        currPage->RemoveFreeMList(freeBlock);
        currPage->RemoveFreeSList(freeBlock);

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
            currPage->InsertFreeMList(newBlock);
            currPage->InsertFreeSList(newBlock);
        }

        // Insert the free block into the used list
        currPage->InsertUsedList(freeBlock);

        // Return the free block address
        *addr = baseAddress_ + freeBlock->addr;

        // Update allocated size
        allocated_ += size;

        return 0;
    }

    int release(uint64_t addr) {
        // Walk all pages to find the pointer
        uint64_t local_addr = addr - baseAddress_;
        block_t* usedBlock = nullptr;
        auto currPage = pages_;
        while (currPage) {
            if (local_addr >= currPage->addr
            &&  local_addr < (currPage->addr + currPage->size)) {
                auto currBlock = currPage->usedList;
                while (currBlock) {
                    if (currBlock->addr == local_addr) {
                        usedBlock = currBlock;
                        break;
                    }
                    currBlock = currBlock->nextUsed;
                }
                break;
            }
            currPage = currPage->next;
        }

        // found the corresponding block?
        if (nullptr == usedBlock) {
            printf("error: invalid address to release: 0x%lx\n", addr);
            return -1;
        }

        auto size = usedBlock->size;

        // Remove the block from the used list
        currPage->RemoveUsedList(usedBlock);

        // Insert the block into the free M-list.
        currPage->InsertFreeMList(usedBlock);

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
                currPage->RemoveFreeSList(prevBlock);

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
                currPage->RemoveFreeSList(nextBlock);
                delete nextBlock;
            }
        }

        // Insert the block into the free S-list.
        currPage->InsertFreeSList(usedBlock);

        // Check if we can free empty pages
        if (nullptr == currPage->usedList) {
            // Try to delete the page
            while (currPage && this->DeletePage(currPage)) {
                currPage = this->FindNextEmptyPage();
            }

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
        
        // List of used blocks
        block_t* usedList;
        
        // List with blocks sorted by descreasing sizes
        // Used for block lookup during memory allocation.
        block_t* freeSList;
        
        // List with blocks sorted by increasing memory addresses
        // Used for block merging during memory release.
        block_t* freeMList;
        
        uint64_t addr;
        uint64_t size;

        page_t(uint64_t addr, uint64_t size) : 
            next(nullptr),            
            usedList(nullptr),
            addr(addr),
            size(size) {
            freeSList = freeMList = new block_t(addr, size);
        }

        void InsertUsedList(block_t* block) {
            block->nextUsed = usedList;
            if (usedList) {
                usedList->prevUsed = block;
            }
            usedList = block;
        }

        void RemoveUsedList(block_t* block) {
            if (block->prevUsed) {
                block->prevUsed->nextUsed = block->nextUsed;
            } else {
                usedList = block->nextUsed;
            }
            if (block->nextUsed) {
                block->nextUsed->prevUsed = block->prevUsed;
            }
            block->nextUsed = nullptr;
            block->prevUsed = nullptr;
        }

        void InsertFreeMList(block_t* block) {
            block_t* currBlock = freeMList;
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
                freeMList = block;
            }
            if (currBlock) {
                currBlock->prevFreeM = block;
            }    
        }

        void RemoveFreeMList(block_t* block) {
            if (block->prevFreeM) {
                block->prevFreeM->nextFreeM = block->nextFreeM;
            } else {
                freeMList = block->nextFreeM;
            }
            if (block->nextFreeM) {
                block->nextFreeM->prevFreeM = block->prevFreeM;
            }
            block->nextFreeM = nullptr;
            block->prevFreeM = nullptr;
        }

        void InsertFreeSList(block_t* block) {
            block_t* currBlock = this->freeSList;
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
                this->freeSList = block;
            }
            if (currBlock) {
                currBlock->prevFreeS = block;
            }
        }

        void RemoveFreeSList(block_t* block) {
            if (block->prevFreeS) {
                block->prevFreeS->nextFreeS = block->nextFreeS;
            } else {
                freeSList = block->nextFreeS;
            }
            if (block->nextFreeS) {
                block->nextFreeS->prevFreeS = block->prevFreeS;
            }
            block->nextFreeS = nullptr;
            block->prevFreeS = nullptr;    
        }
    };

    page_t* NewPage(uint64_t size) {
        // Increase buffer size to include the page and first block size
        // also add padding to ensure page alignment
        size = AlignSize(size, pageAlign_);

        // Allocate page memory
        auto addr = nextAddress_;
        nextAddress_ += size;

        // Overflow check
        if (nextAddress_ > capacity_)
            return nullptr;

        // Allocate object
        auto newPage = new page_t(addr, size);

        // Insert the new page into the list
        newPage->next = pages_;
        pages_ = newPage;

        return newPage;
    }

    bool DeletePage(page_t* page) {
        // The page should be empty
        assert(nullptr == page->usedList);
        assert(page->freeMList && (nullptr == page->freeMList->nextFreeM));

        // Only delete top-level pages
        auto nextAddr = page->addr + page->size;
        if (nextAddr != nextAddress_)
            return false;

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

        // Update next allocation address
        nextAddress_ = page->addr;
        
        // free object
        delete page->freeMList;
        delete page;

        return true;
    }

    page_t* FindNextEmptyPage() {
       auto currPage = pages_;
        while (currPage) {
            if (nullptr == currPage->usedList)
                return currPage;
            currPage = currPage->next;
        } 
        return nullptr;
    }

    static uint64_t AlignSize(uint64_t size, uint64_t alignment) {
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