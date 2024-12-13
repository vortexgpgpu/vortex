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
#include <functional>

#ifndef VM_ENABLE
#define VM_ENABLE
#endif

using namespace vortex;

class VMManager {
public:
    VMManager(vx_device* device)
        : device_(device)
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

        if (device_->mem_reserve(PAGE_TABLE_BASE_ADDR, PT_SIZE_LIMIT, VX_MEM_READ_WRITE) != 0) {
            std::cerr << "Failed to reserve space for Page Table\n";
            return 1;
        }

        page_table_mem_ = new MemoryAllocator(PAGE_TABLE_BASE_ADDR, PT_SIZE_LIMIT, MEM_PAGE_SIZE, CACHE_BLOCK_SIZE);
        if (!page_table_mem_) {
            std::cerr << "Failed to initialize page_table_mem_\n";
            device_->mem_free(PAGE_TABLE_BASE_ADDR);
            return 1;
        }

        virtual_mem_ = new MemoryAllocator(ALLOC_BASE_ADDR, (GLOBAL_MEM_SIZE - ALLOC_BASE_ADDR), MEM_PAGE_SIZE, CACHE_BLOCK_SIZE);
        CHECK_ERR(virtual_mem_reserve(PAGE_TABLE_BASE_ADDR, (GLOBAL_MEM_SIZE - PAGE_TABLE_BASE_ADDR), VX_MEM_READ_WRITE), {
            return err;
        });
        CHECK_ERR(virtual_mem_reserve(STARTUP_ADDR, 0x40000, VX_MEM_READ_WRITE), {
            return err;
        });
        
        if (VM_ADDR_MODE != BARE && alloc_page_table(&pt_addr) != 0) {
            std::cerr << "Failed to allocate page table\n";
            return 1;
        }

        if (device_->set_satp(pt_addr) != 0) {
            std::cerr << "Failed to set SATP register\n";
            return 1;
        }

        return 0;
    }

    uint64_t map_p2v(uint64_t ppn, uint32_t flags) {
        if (ppn2vpn.find(ppn) != ppn2vpn.end()) return ppn2vpn[ppn];

        uint64_t vpn;
        virtual_mem_->allocate(MEM_PAGE_SIZE, &vpn);
        vpn >>= MEM_PAGE_LOG2_SIZE;

        if (update_page_table(ppn, vpn, flags) != 0) {
            throw std::runtime_error("Failed to update page table");
        }

        ppn2vpn[ppn] = vpn;
        return vpn;
    }

    bool need_trans(uint64_t dev_pAddr) {
        if (device_->is_satp_unset() || get_mode() == BARE) return false;
        if (PAGE_TABLE_BASE_ADDR <= dev_pAddr) return false;
        if (dev_pAddr < USER_BASE_ADDR) return false;
        return !(STARTUP_ADDR <= dev_pAddr && dev_pAddr <= (STARTUP_ADDR + 0x40000));
    }

    uint64_t virt_to_phys_map(uint64_t vAddr) {
        return va2pa[vAddr];
    }

    uint64_t phys_to_virt_map(uint64_t* dev_vAddr, uint64_t size, const uint64_t* dev_pAddr, uint32_t flags) {
        if (!need_trans(*dev_pAddr)) return 0;

        uint64_t init_pAddr = *dev_pAddr;
        uint64_t init_vAddr = (map_p2v(init_pAddr >> MEM_PAGE_LOG2_SIZE, flags) << MEM_PAGE_LOG2_SIZE) |
                              (init_pAddr & ((1 << MEM_PAGE_LOG2_SIZE) - 1));

        for (uint64_t ppn = (*dev_pAddr >> MEM_PAGE_LOG2_SIZE);
             ppn < ((*dev_pAddr) >> MEM_PAGE_LOG2_SIZE) + (size >> MEM_PAGE_LOG2_SIZE);
             ++ppn) {
            map_p2v(ppn, flags);
        }

        *dev_vAddr = init_vAddr;
        va2pa[init_vAddr] = init_pAddr;
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

    // uint64_t page_table_walk(uint64_t vAddr_bits) {
    //     if (!need_trans(vAddr_bits)) return vAddr_bits;

    //     vAddr_t vaddr(vAddr_bits);
    //     uint64_t cur_base_ppn = get_base_ppn();
    //     uint64_t pte_addr = 0, pte_bytes = 0;
    //     int i = PT_LEVEL - 1;

    //     while (true) {
    //         pte_addr = get_pte_address(cur_base_ppn, vaddr.vpn[i]);
    //         pte_bytes = read_pte(pte_addr);
    //         PTE_t pte(pte_bytes);

    //         assert(((pte.pte_bytes & 0xFFFFFFFF) != 0xbaadf00d) && "ERROR: uninitialzed PTE\n" );
    //         // Check if it has invalid flag bits.
    //         if ((pte.v == 0) | ((pte.r == 0) & (pte.w == 1)))
    //         {
    //             std::string msg = "  [RT:PTW] Page Fault : Attempted to access invalid entry.";
    //             throw Page_Fault_Exception(msg);
    //         }

    //         if ((pte.r == 0) & (pte.w == 0) & (pte.x == 0))
    //         {
    //             i--;
    //             // Not a leaf node as rwx == 000
    //             if (i < 0)
    //             {
    //                 throw Page_Fault_Exception("  [RT:PTW] Page Fault : No leaf node found.");
    //             }
    //             else
    //             {
    //                 // Continue on to next level.
    //                 cur_base_ppn = pte.ppn;
    //                 std::cout << "  [RT:PTW] next base_ppn: 0x" << cur_base_ppn << std::endl;
    //                 continue;
    //             }
    //         }
    //         else
    //         {
    //             // Leaf node found. 
    //             // Check RWX permissions according to access type.
    //             if (pte.r == 0)
    //             {
    //                 throw Page_Fault_Exception("  [RT:PTW] Page Fault : TYPE LOAD, Incorrect permissions.");
    //             }
    //             cur_base_ppn = pte.ppn;
    //             std::cout << "  [RT:PTW] Found PT_Base_Address(0x" << cur_base_ppn << ") on Level " << i << std::endl;
    //             break;
    //         }
    //     }

    //     uint64_t paddr = (cur_base_ppn << MEM_PAGE_LOG2_SIZE) + vaddr.pgoff;
    //     return paddr;
    // }

private:
    uint64_t get_base_ppn() {
        return device_->get_base_ppn();
    }

    uint64_t get_pte_address(uint64_t base_ppn, uint64_t vpn) {
        return (base_ppn * PT_SIZE) + (vpn * PTE_SIZE);
    }

    uint64_t read_pte(uint64_t addr) {
        uint64_t value = 0;
        device_->download(&value, addr, sizeof(uint64_t));
        return value;
    }

    void write_pte(uint64_t addr, uint64_t value) {
        device_->upload(&value, addr, sizeof(uint64_t));
    }

    // Initialize to zero the target page table area. 32bit 4K, 64bit 8K
    uint16_t init_page_table(uint64_t addr, uint64_t size)
    {
        uint64_t asize = aligned_size(size, CACHE_BLOCK_SIZE);
        uint8_t *src = new uint8_t[asize];
        if (src == nullptr)
            return 1;

        for (uint64_t i = 0; i < asize; ++i)
        {
            src[i] = 0;
        }
        device_->upload(src, addr, asize);
        return 0;
    }

    int alloc_page_table(uint64_t* pt_addr) {
        if (page_table_mem_->allocate(PT_SIZE, pt_addr) != 0) return 1;
        if (init_page_table(*pt_addr, PT_SIZE) != 0) return 1;
        return 0;
    }

    int virtual_mem_reserve(uint64_t dev_addr, uint64_t size)
    {
        if (virtual_mem_->reserve(dev_addr, size) != 0) return 1;
        return 0;
    }

    uint8_t get_mode() {
        return device_->get_satp_mode();
    }

private:
    vx_device* device_;                               // to set SATP, write ptes
    MemoryAllocator* page_table_mem_;                 // allocate PT memory
    MemoryAllocator* virtual_mem_;                    // allocate virtual memory
    std::unordered_map<uint64_t, uint64_t> ppn2vpn;   // ppn to vpn mapping
    std::unordered_map<uint64_t, uint64_t> va2pa;     // vAddr to pAddr mapping
};