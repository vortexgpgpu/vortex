#include <common.h>

#include <unordered_map>
#include <VX_config.h>
#include <cassert>
#include <iostream>
#include <mem.h>
#include <processor.h>
#include <stdint.h>
#include <cstdlib>
#include <cmath>

#include <util.h>
#include <vortex.h>

#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <future>
#include <chrono>

#ifndef VM_ENABLE
#define VM_ENABLE
#endif

using namespace vortex;

class VMManager {
public:
    VMManager(Processor& processor, MemoryAllocator& global_mem, RAM& ram)
        : processor_(processor)
        , global_mem_(global_mem)
        , ram_(ram)
    {
        page_table_mem_ = nullptr;
        virtual_mem_ = nullptr;
    }

    ~VMManager() {
        if (page_table_mem_) delete page_table_mem_;
        if (virtual_mem_) delete virtual_mem_;
    }

    int16_t init_VM() {
        uint64_t pt_addr = 0;

        // Reserve space for Page Table
        std::cout << "[VMManager:init_VM] Initializing VM\n";
        std::cout << "* PAGE_TABLE_BASE_ADDR=" << std::hex << PAGE_TABLE_BASE_ADDR << "\n";

        if (global_mem_.reserve(PAGE_TABLE_BASE_ADDR, PT_SIZE_LIMIT) != 0) {
            std::cerr << "Failed to reserve space for Page Table\n";
            return 1;
        }

        page_table_mem_ = new MemoryAllocator(PAGE_TABLE_BASE_ADDR, PT_SIZE_LIMIT, MEM_PAGE_SIZE, CACHE_BLOCK_SIZE);
        if (!page_table_mem_) {
            std::cerr << "Failed to initialize page_table_mem_\n";
            global_mem_.release(PAGE_TABLE_BASE_ADDR);
            return 1;
        }

        virtual_mem_ = new MemoryAllocator(ALLOC_BASE_ADDR, GLOBAL_MEM_SIZE - ALLOC_BASE_ADDR, MEM_PAGE_SIZE, CACHE_BLOCK_SIZE);
        if (!virtual_mem_) {
            std::cerr << "Failed to initialize virtual_mem_\n";
            return 1;
        }

        if (processor_.set_satp_by_addr(pt_addr) != 0) {
            std::cerr << "Failed to set SATP register\n";
            return 1;
        }

        return 0;
    }

    uint64_t map_p2v(uint64_t ppn, uint32_t flags) {
        if (addr_mapping.find(ppn) != addr_mapping.end()) return addr_mapping[ppn];

        uint64_t vpn;
        virtual_mem_->allocate(MEM_PAGE_SIZE, &vpn);
        vpn >>= MEM_PAGE_LOG2_SIZE;

        if (update_page_table(ppn, vpn, flags) != 0) {
            throw std::runtime_error("Failed to update page table");
        }

        addr_mapping[ppn] = vpn;
        return vpn;
    }

    bool need_trans(uint64_t dev_pAddr) {
        if (processor_.is_satp_unset() || get_mode() == BARE) return false;
        if (PAGE_TABLE_BASE_ADDR <= dev_pAddr) return false;
        if (dev_pAddr < USER_BASE_ADDR) return false;
        return !(STARTUP_ADDR <= dev_pAddr && dev_pAddr <= (STARTUP_ADDR + 0x40000));
    }

    uint64_t phy_to_virt_map(uint64_t size, uint64_t* dev_pAddr, uint32_t flags) {
        if (!need_trans(*dev_pAddr)) return 0;

        uint64_t init_pAddr = *dev_pAddr;
        uint64_t init_vAddr = (map_p2v(init_pAddr >> MEM_PAGE_LOG2_SIZE, flags) << MEM_PAGE_LOG2_SIZE) |
                              (init_pAddr & ((1 << MEM_PAGE_LOG2_SIZE) - 1));

        for (uint64_t ppn = (*dev_pAddr >> MEM_PAGE_LOG2_SIZE);
             ppn < ((*dev_pAddr) >> MEM_PAGE_LOG2_SIZE) + (size >> MEM_PAGE_LOG2_SIZE);
             ++ppn) {
            map_p2v(ppn, flags);
        }

        *dev_pAddr = init_vAddr;
        return 0;
    }

    int16_t update_page_table(uint64_t ppn, uint64_t vpn, uint32_t flags) {
        uint64_t cur_base_ppn = get_base_ppn();
        int i = PT_LEVEL - 1;

        while (i >= 0) {
            uint64_t pte_addr = get_pte_address(cur_base_ppn, vpn >> (i * MEM_PAGE_LOG2_SIZE));
            uint64_t pte = read_pte(pte_addr);

            if (pte & 1) {
                cur_base_ppn = pte >> MEM_PAGE_LOG2_SIZE;
            } else {
                if (i == 0) {
                    write_pte(pte_addr, (ppn << MEM_PAGE_LOG2_SIZE) | flags);
                } else {
                    uint64_t next_pt;
                    if (alloc_page_table(&next_pt) != 0) return 1;
                    write_pte(pte_addr, (next_pt << MEM_PAGE_LOG2_SIZE) | 1);
                    cur_base_ppn = next_pt >> MEM_PAGE_LOG2_SIZE;
                }
            }
            i--;
        }

        return 0;
    }

    uint64_t page_table_walk(uint64_t vAddr) {
        if (!need_trans(vAddr)) return vAddr;

        uint64_t cur_base_ppn = get_base_ppn();
        int i = PT_LEVEL - 1;

        while (true) {
            uint64_t pte_addr = get_pte_address(cur_base_ppn, vAddr >> (i * MEM_PAGE_LOG2_SIZE));
            uint64_t pte = read_pte(pte_addr);

            if (pte & 1) {
                if (i == 0) return (pte >> MEM_PAGE_LOG2_SIZE) | (vAddr & (MEM_PAGE_SIZE - 1));
                cur_base_ppn = pte >> MEM_PAGE_LOG2_SIZE;
                i--;
            } else {
                throw std::runtime_error("Page fault during page table walk");
            }
        }
    }

private:
    uint64_t get_base_ppn() {
        return processor_.get_base_ppn();
    }

    uint64_t get_pte_address(uint64_t base_ppn, uint64_t vpn) {
        return (base_ppn << MEM_PAGE_LOG2_SIZE) + (vpn * sizeof(uint64_t));
    }

    uint64_t read_pte(uint64_t addr) {
        uint64_t value = 0;
        ram_.read(reinterpret_cast<uint8_t*>(&value), addr, sizeof(uint64_t));
        return value;
    }

    void write_pte(uint64_t addr, uint64_t value) {
        ram_.write(reinterpret_cast<const uint8_t*>(&value), addr, sizeof(uint64_t));
    }

    int alloc_page_table(uint64_t* pt_addr) {
        if (page_table_mem_->allocate(PT_SIZE, pt_addr) != 0) return 1;
        uint8_t zero[PT_SIZE] = {0};
        ram_.write(zero, *pt_addr, PT_SIZE);
        return 0;
    }

    uint8_t get_mode() {
        return processor_.get_satp_mode();
    }

private:
    Processor& processor_;
    RAM& ram_;
    MemoryAllocator& global_mem_;
    MemoryAllocator* page_table_mem_;
    MemoryAllocator* virtual_mem_;
    std::unordered_map<uint64_t, uint64_t> addr_mapping;
};