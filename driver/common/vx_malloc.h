#pragma once

#include <cstdint>
#include <assert.h>

namespace vortex {

class MemoryAllocator {
public:
    MemoryAllocator(
        uint64_t minAddress,
        uint64_t maxAddress,
        uint32_t pageAlign, 
        uint32_t blockAlign) 
        : nextAddress_(minAddress)
        , maxAddress_(maxAddress)
        , pageAlign_(pageAlign)
        , blockAlign_(blockAlign)
        , pages_(nullptr)
    {}

    ~MemoryAllocator() {
        // Free allocated pages
        page_t* pCurPage = pages_;
        while (pCurPage) {
            auto nextPage = pCurPage->next;
            this->DeletePage(pCurPage);
            pCurPage = nextPage;
        }
    }

    int allocate(uint64_t size, uint64_t* addr) {
        if (size == 0 || addr == nullptr)
            return -1;

        // Align allocation size
        size = AlignSize(size, blockAlign_);

        // Walk thru all pages to find a free block
        block_t* pFreeBlock = nullptr;
        auto pCurPage = pages_;
        while (pCurPage) {
            auto pCurBlock = pCurPage->pFreeSList;
            if (pCurBlock) {
                // The free list is already sorted with biggest block on top,
                // just check if the last block has enough space.
                if (pCurBlock->size >= size) {
                    // Find the smallest matching block
                    while (pCurBlock->nextFreeS 
                        && (pCurBlock->nextFreeS->size >= size)) {
                        pCurBlock = pCurBlock->nextFreeS;
                    }
                    // Return the free block
                    pFreeBlock = pCurBlock;
                    break;
                }
            }
            pCurPage = pCurPage->next;
        }

        if (nullptr == pFreeBlock) {
            // Allocate a new page for this request
            pCurPage = this->NewPage(size);
            if (nullptr == pCurPage)
                return -1;
            pFreeBlock = pCurPage->pFreeSList;
        }   

        // Remove the block from the free lists
        assert(pFreeBlock->size >= size);
        pCurPage->RemoveFreeMBlock(pFreeBlock);
        pCurPage->RemoveFreeSBlock(pFreeBlock);

        // If the free block we have found is larger than what we are looking for,
        // we may be able to split our free block in two.
        uint64_t extraBytes = pFreeBlock->size - size;
        if (extraBytes >= blockAlign_) {
            // Reduce the free block size to the requested value
            pFreeBlock->size = size;

            // Allocate a new block to contain the extra buffer
            auto nextAddr = pFreeBlock->addr + size;
            auto pNewBlock = new block_t(nextAddr, extraBytes);

            // Add the new block to the free lists
            pCurPage->InsertFreeMBlock(pNewBlock);
            pCurPage->InsertFreeSBlock(pNewBlock);
        }

        // Insert the free block into the used list
        pCurPage->InsertUsedBlock(pFreeBlock);

        // Return the free block address
        *addr = pFreeBlock->addr;

        return 0;
    }

    int release(uint64_t addr) {
        // Walk all pages to find the pointer
        block_t* pUsedBlock = nullptr;
        auto pCurPage = pages_;
        while (pCurPage) {
            if ((pCurPage->addr < addr)
            && ((pCurPage->addr + pCurPage->size) > addr)) {
                auto pCurBlock = pCurPage->pUsedList;
                while (pCurBlock) {
                    if (pCurBlock->addr == addr) {
                        pUsedBlock = pCurBlock;
                        break;
                    }
                    pCurBlock = pCurBlock->nextUsed;
                }
                if (pUsedBlock)
                    break;
            }
            pCurPage = pCurPage->next;
        }

        // found the corresponding block?
        if (nullptr == pUsedBlock)
            return -1;

        // Remove the block from the used list
        pCurPage->RemoveUsedBlock(pUsedBlock);

        // Insert the block into the free M-list.
        pCurPage->InsertFreeMBlock(pUsedBlock);

        // Check if we can merge adjacent free blocks from the left.        
        if (pUsedBlock->prevFreeM) {
            // Calculate the previous address
            auto prevAddr = pUsedBlock->prevFreeM->addr + pUsedBlock->prevFreeM->size;
            if (pUsedBlock->addr == prevAddr) {
                auto pMergedBlock = pUsedBlock->prevFreeM;

                // Detach left block from the free S-list
                pCurPage->RemoveFreeSBlock(pMergedBlock);

                // Merge the blocks to the left
                pMergedBlock->size += pUsedBlock->size;
                pMergedBlock->nextFreeM = pUsedBlock->nextFreeM;
                if (pMergedBlock->nextFreeM) {
                    pMergedBlock->nextFreeM->prevFreeM = pMergedBlock;
                }
                pUsedBlock = pMergedBlock;
            }
        }

        // Check if we can merge adjacent free blocks from the right.
        if (pUsedBlock->nextFreeM) {
            // Calculate the next allocation start address
            auto nextMem = pUsedBlock->addr + pUsedBlock->size;
            if (pUsedBlock->nextFreeM->addr == nextMem) {
                auto nextBlock = pUsedBlock->nextFreeM;

                // Detach right block from the free S-list
                pCurPage->RemoveFreeSBlock(nextBlock);

                // Merge the blocks to the right
                pUsedBlock->size += nextBlock->size;
                pUsedBlock->nextFreeM = nextBlock->nextFreeM;
                if (pUsedBlock->nextFreeM) {
                    pUsedBlock->nextFreeM->prevFreeM = pUsedBlock;
                }
            }
        }

        // Insert the block into the free S-list.
        pCurPage->InsertFreeSBlock(pUsedBlock);

        // Check if we can free empty pages
        if (nullptr == pCurPage->pUsedList) {
            // Try to delete the page
            while (pCurPage && this->DeletePage(pCurPage)) {
                pCurPage = this->NextEmptyPage();
            }

        }

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
        block_t* pUsedList;
        
        // List with blocks sorted by descreasing sizes
        // Used for block lookup during memory allocation.
        block_t* pFreeSList;
        
        // List with blocks sorted by increasing memory addresses
        // Used for block merging during memory release.
        block_t* pFreeMList;
        
        uint64_t addr;
        uint64_t size;

        page_t(uint64_t addr, uint64_t size) : 
            next(nullptr),            
            pUsedList(nullptr),
            addr(addr),
            size(size) {
            pFreeSList = pFreeMList = new block_t(addr, size);
        }

        void InsertUsedBlock(block_t* pBlock) {
            pBlock->nextUsed = pUsedList;
            if (pUsedList) {
                pUsedList->prevUsed = pBlock;
            }
            pUsedList = pBlock;
        }

        void RemoveUsedBlock(block_t* pBlock) {
            if (pBlock->prevUsed) {
                pBlock->prevUsed->nextUsed = pBlock->nextUsed;
            } else {
                pUsedList = pBlock->nextUsed;
            }
            if (pBlock->nextUsed) {
                pBlock->nextUsed->prevUsed = pBlock->prevUsed;
            }
            pBlock->nextUsed = nullptr;
            pBlock->prevUsed = nullptr;
        }

        void InsertFreeMBlock(block_t* pBlock) {
            block_t* pCurBlock = pFreeMList;
            block_t* prevBlock = nullptr;
            while (pCurBlock && (pCurBlock->addr < pBlock->addr)) {
                prevBlock = pCurBlock;
                pCurBlock = pCurBlock->nextFreeM;
            }
            pBlock->nextFreeM = pCurBlock;
            pBlock->prevFreeM = prevBlock;
            if (prevBlock) {
                prevBlock->nextFreeM = pBlock;
            } else {
                pFreeMList = pBlock;
            }
            if (pCurBlock) {
                pCurBlock->prevFreeM = pBlock;
            }    
        }

        void RemoveFreeMBlock(block_t* pBlock) {
            if (pBlock->prevFreeM) {
                pBlock->prevFreeM->nextFreeM = pBlock->nextFreeM;
            } else {
                pFreeMList = pBlock->nextFreeM;
            }
            if (pBlock->nextFreeM) {
                pBlock->nextFreeM->prevFreeM = pBlock->prevFreeM;
            }
            pBlock->nextFreeM = nullptr;
            pBlock->prevFreeM = nullptr;
        }

        void InsertFreeSBlock(block_t* pBlock) {
            block_t* pCurBlock = this->pFreeSList;
            block_t* prevBlock = nullptr;
            while (pCurBlock && (pCurBlock->size > pBlock->size)) {
                prevBlock = pCurBlock;
                pCurBlock = pCurBlock->nextFreeS;
            }
            pBlock->nextFreeS = pCurBlock;
            pBlock->prevFreeS = prevBlock;
            if (prevBlock) {
                prevBlock->nextFreeS = pBlock;
            } else {
                this->pFreeSList = pBlock;
            }
            if (pCurBlock) {
                pCurBlock->prevFreeS = pBlock;
            }
        }

        void RemoveFreeSBlock(block_t* pBlock) {
            if (pBlock->prevFreeS) {
                pBlock->prevFreeS->nextFreeS = pBlock->nextFreeS;
            } else {
                pFreeSList = pBlock->nextFreeS;
            }
            if (pBlock->nextFreeS) {
                pBlock->nextFreeS->prevFreeS = pBlock->prevFreeS;
            }
            pBlock->nextFreeS = nullptr;
            pBlock->prevFreeS = nullptr;    
        }
    };

    page_t* NewPage(uint64_t size) {
        // Increase buffer size to include the page and first block size
        // also add padding to ensure page aligment
        size = AlignSize(size, pageAlign_);

        // Allocate page memory
        auto addr = nextAddress_;
        nextAddress_ += size;

        // Overflow check
        if (nextAddress_ > maxAddress_)
            return nullptr;

        // Allocate the page
        auto pNewPage = new page_t(addr, size);

        // Insert the new page into the list
        pNewPage->next = pages_;
        pages_ = pNewPage;

        return pNewPage;
    }

    bool DeletePage(page_t* pPage) {
        // The page should be empty
        assert(nullptr == pPage->pUsedList);
        assert(pPage->pFreeMList && (nullptr == pPage->pFreeMList->nextFreeM));

        // Only delete top-level pages
        auto nextAddr = pPage->addr + pPage->size;
        if (nextAddr != nextAddress_)
            return false;

        // Remove the page from the list
        page_t* prevPage = nullptr;
        auto pCurPage = pages_;
        while (pCurPage) {
            if (pCurPage == pPage) {
                if (prevPage) {
                    prevPage->next = pCurPage->next;
                } else {
                    pages_ = pCurPage->next;
                }
                break;
            }
            prevPage = pCurPage;
            pCurPage = pCurPage->next;
        }

        // Update next allocation address
        nextAddress_ = pPage->addr;

        return true;
    }

    page_t* NextEmptyPage() {
       auto pCurPage = pages_;
        while (pCurPage) {
            if (nullptr == pCurPage->pUsedList)
                return pCurPage;
            pCurPage = pCurPage->next;
        } 
        return nullptr;
    }

    static uint64_t AlignSize(uint64_t size, uint64_t alignment) {
        assert(0 == (alignment & (alignment - 1)));
        return (size + alignment - 1) & ~(alignment - 1);
    }

    uint64_t nextAddress_;
    uint64_t maxAddress_;
    uint32_t pageAlign_;    
    uint32_t blockAlign_;
    page_t* pages_;    
};

} // namespace vortex