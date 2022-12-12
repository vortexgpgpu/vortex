#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <iostream>
#include <future>
#include <chrono>
#include <bitset>

#include <vortex.h>
#include <vx_utils.h>
#include <vx_malloc.h>

#include <VX_config.h>

#include <util.h>

#include <processor.h>
#include <archdef.h>
#include <mem.h>
#include <constants.h>

uint64_t bits(uint64_t addr, uint8_t s_idx, uint8_t e_idx)
{
    return (addr >> s_idx) & ((1 << (e_idx - s_idx + 1)) - 1);
}
bool bit(uint64_t addr, uint8_t idx)
{
    return (addr) & (1 << idx);
}
using namespace vortex;

///////////////////////////////////////////////////////////////////////////////

class vx_device;

class vx_buffer {
public:
    vx_buffer(uint64_t size, vx_device* device) 
        : size_(size)
        , device_(device) {
        uint64_t aligned_asize = aligned_size(size, CACHE_BLOCK_SIZE);
        data_ = malloc(aligned_asize);
    }

    ~vx_buffer() {
        if (data_) {
            free(data_);
        }
    }

    void* data() const {
        return data_;
    }

    uint64_t size() const {
        return size_;
    }

    vx_device* device() const {
        return device_;
    }

private:
    uint64_t size_;
    vx_device* device_;
    void* data_;
};

///////////////////////////////////////////////////////////////////////////////

class vx_device {    
public:
    vx_device() 
        : arch_(NUM_CORES * NUM_CLUSTERS, NUM_WARPS, NUM_THREADS)
        , ram_(RAM_PAGE_SIZE)
        , processor_(arch_)
        , mem_allocator_(
            ALLOC_BASE_ADDR,
            ALLOC_BASE_ADDR + LOCAL_MEM_SIZE,
            RAM_PAGE_SIZE,
            CACHE_BLOCK_SIZE) 
    {
        processor_.attach_ram(&ram_);
        //Sets more
        set_processor_satp(VM_ADDR_MODE);
    }

    ~vx_device() {
        if (future_.valid()) {
            future_.wait();
        }
    }    
    
    int map_local_mem(uint64_t size, uint64_t dev_maddr) 
    {
        if (get_mode() == VA_MODE::BARE)
            return 0;

        uint32_t ppn = dev_maddr >> 12;
        uint32_t vpn = ppn;

        //dev_maddr can be of size greater than a page, but we have to map and update
        //page tables on a page table granularity. So divide the allocation into pages.
        for (ppn = (dev_maddr) >> 12; ppn < ((dev_maddr) >> 12) + (size/RAM_PAGE_SIZE) + 1; ppn++)
        {
            //Currently a 1-1 mapping is used, this can be changed here to support different
            //mapping schemes
            vpn = ppn;

            //If ppn to vpn mapping doesnt exist.
            if (addr_mapping.find(vpn) == addr_mapping.end())
            {
                //Create mapping.
                update_page_table(ppn, vpn);
                addr_mapping[vpn] = ppn;
            }
        }
        return 0;
    }

    int alloc_local_mem(uint64_t size, uint64_t* dev_maddr) {
        int err = mem_allocator_.allocate(size, dev_maddr);
        map_local_mem(size, *dev_maddr);
        return err;
    }

    int free_local_mem(uint64_t dev_maddr) {
        return mem_allocator_.release(dev_maddr);
    }

    int upload(const void* src, uint64_t dest_addr, uint64_t size, uint64_t src_offset) {
        uint64_t asize = aligned_size(size, CACHE_BLOCK_SIZE);
        if (dest_addr + asize > LOCAL_MEM_SIZE)
            return -1;

        if (dest_addr >= STARTUP_ADDR)
            map_local_mem(asize,dest_addr);
        else if (dest_addr >= 0x7fff0000)
        {
            map_local_mem(asize,dest_addr);
        }
        ram_.write((const uint8_t*)src + src_offset, dest_addr, asize);        
        return 0;
    }

    int download(void* dest, uint64_t src_addr, uint64_t size, uint64_t dest_offset) {
        uint64_t asize = aligned_size(size, CACHE_BLOCK_SIZE);
        if (src_addr + asize > LOCAL_MEM_SIZE)
            return -1;

        ram_.read((uint8_t*)dest + dest_offset, src_addr, asize);
        
        /*printf("VXDRV: download %d bytes from 0x%x\n", size, src_addr);
        for (int i = 0; i < size; i += 4) {
            printf("mem-read: 0x%x -> 0x%x\n", src_addr + i, *(uint32_t*)((uint8_t*)dest + dest_offset + i));
        }*/
        
        return 0;
    }

    int start() {  
        // ensure prior run completed
        if (future_.valid()) {
            future_.wait();
        }
        
        // start new run
        future_ = std::async(std::launch::async, [&]{
            processor_.run();
        });
        
        return 0;
    }

    int wait(uint64_t timeout) {
        if (!future_.valid())
            return 0;
        uint64_t timeout_sec = timeout / 1000;
        std::chrono::seconds wait_time(1);
        for (;;) {
            // wait for 1 sec and check status
            auto status = future_.wait_for(wait_time);
            if (status == std::future_status::ready 
             || 0 == timeout_sec--)
                break;
        }
        return 0;
    }

    void set_processor_satp(VA_MODE mode)
    {
        uint32_t satp;
        if (mode == VA_MODE::BARE)
            satp = 0;
        else if (mode == VA_MODE::SV32)
        {
            satp = (alloc_page_table() >> 10) | 0x80000000;
        }
        processor_.set_satp(satp);
    }

    uint32_t get_ptbr()
    {        

        return processor_.get_satp() & 0x003fffff;
    }

    VA_MODE get_mode()
    {
        return processor_.get_satp() & 0x80000000 ? VA_MODE::SV32 : VA_MODE::BARE;
    }  

    void update_page_table(uint32_t pAddr, uint32_t vAddr) {
        //Updating page table with the following mapping of (vAddr) to (pAddr).
        uint32_t ppn_0, ppn_1, pte_addr, pte_bytes;
        uint32_t vpn_1 = bits(vAddr, 10, 19);
        uint32_t vpn_0 = bits(vAddr, 0, 9);

        //Read first level PTE.
        pte_addr = (get_ptbr() << 12) + (vpn_1 * PTE_SIZE);
        pte_bytes = read_pte(pte_addr);        


        if ( bit(pte_bytes, 0) )
        {
            //If valid bit set, proceed to next level using new ppn form PTE.
            ppn_1 = (pte_bytes >> 10);
        }
        else
        {
            //If valid bit not set, allocate a second level page table
            // in device memory and store ppn in PTE. Set rwx = 000 in PTE
            //to indicate this is a pointer to the next level of the page table.
            ppn_1 = (alloc_page_table() >> 12);
            pte_bytes = ( (ppn_1 << 10) | 0b0000000001) ;
            write_pte(pte_addr, pte_bytes);
        }

        //Read second level PTE.
        pte_addr = (ppn_1 << 12) + (vpn_0 * PTE_SIZE);
        pte_bytes = read_pte(pte_addr);        
    
        if ( bit(pte_bytes, 0) )
        {
            //If valid bit is set, then the page is already allocated.
            //Should not reach this point, a sanity check.
        }
        else
        {
            //If valid bit not set, write ppn of pAddr in PTE. Set rwx = 111 in PTE
            //to indicate this is a leaf PTE and has the stated permissions.
            pte_bytes = ( (pAddr << 10) | 0b0000001111) ;
            write_pte(pte_addr, pte_bytes);

            //If super paging is enabled.
            if (SUPER_PAGING)
            {
                //Check if this second level Page Table can be promoted to a super page. Brute force 
                //method is used to iterate over all PTE entries of the table and check if they have 
                //their valid bit set.
                bool superpage = true;
                for(int i = 0; i < 1024; i++)
                {
                    pte_addr = (ppn_1 << 12) + (i * PTE_SIZE);
                    pte_bytes = read_pte(pte_addr); 
                  
                    if (!bit(pte_bytes, 0))
                    {
                        superpage = false;
                        break;
                    }
                }
                if (superpage)
                {
                    //This can be promoted to a super page. Set root PTE to the first PTE of the 
                    //second level. This is because the first PTE of the second level already has the
                    //correct PPN1, PPN0 set to zero and correct access bits.
                    pte_addr = (ppn_1 << 12);
                    pte_bytes = read_pte(pte_addr);
                    pte_addr = (get_ptbr() << 12) + (vpn_1 * PTE_SIZE);
                    write_pte(pte_addr, pte_bytes);
                }
            }
        }
    }

    uint32_t alloc_page_table() {
        uint64_t addr;
        mem_allocator_.allocate(RAM_PAGE_SIZE, &addr);
        init_page_table(addr);
        return addr;
    }


    void init_page_table(uint32_t addr) {
        uint64_t asize = aligned_size(RAM_PAGE_SIZE, CACHE_BLOCK_SIZE);
        uint8_t *src = new uint8_t[RAM_PAGE_SIZE];
        for (uint32_t i = 0; i < RAM_PAGE_SIZE; ++i) {
            src[i] = (0x00000000 >> ((i & 0x3) * 8)) & 0xff;
        }
        ram_.write((const uint8_t*)src, addr, asize);
    }

    void read_page_table(uint32_t addr) {
        uint8_t *dest = new uint8_t[RAM_PAGE_SIZE];
        download(dest,  addr,  RAM_PAGE_SIZE, 0);
        printf("VXDRV: download %d bytes from 0x%x\n", RAM_PAGE_SIZE, addr);
        for (int i = 0; i < RAM_PAGE_SIZE; i += 4) {
            printf("mem-read: 0x%x -> 0x%x\n", addr + i, *(uint32_t*)((uint8_t*)dest + i));
        }
    }

    void write_pte(uint32_t addr, uint32_t value = 0xbaadf00d) {
        uint8_t *src = new uint8_t[PTE_SIZE];
        for (uint32_t i = 0; i < PTE_SIZE; ++i) {
            src[i] = (value >> ((i & 0x3) * 8)) & 0xff;
        }
        ram_.write((const uint8_t*)src, addr, PTE_SIZE);
    }

    uint32_t read_pte(uint32_t addr) {
        uint8_t *dest = new uint8_t[PTE_SIZE];
        ram_.read((uint8_t*)dest, addr, PTE_SIZE);
        return *(uint32_t*)((uint8_t*)dest);
    }
    
private:
    ArchDef arch_;
    RAM ram_;
    Processor processor_;
    MemoryAllocator mem_allocator_;       
    std::future<void> future_;
    std::unordered_map<uint32_t, uint32_t> addr_mapping;
};

///////////////////////////////////////////////////////////////////////////////

#ifdef DUMP_PERF_STATS
class AutoPerfDump {
private:
    std::list<vx_device_h> devices_;

public:
    AutoPerfDump() {} 

    ~AutoPerfDump() {
        for (auto device : devices_) {
            vx_dump_perf(device, stdout);
        }
    }

    void add_device(vx_device_h device) {
        devices_.push_back(device);
    }

    void remove_device(vx_device_h device) {
        devices_.remove(device);
    }    
};

AutoPerfDump gAutoPerfDump;
#endif

///////////////////////////////////////////////////////////////////////////////

extern int vx_dev_open(vx_device_h* hdevice) {
    if (nullptr == hdevice)
        return  -1;

    *hdevice = new vx_device();    

#ifdef DUMP_PERF_STATS
    gAutoPerfDump.add_device(*hdevice);
#endif

    return 0;
}

extern int vx_dev_close(vx_device_h hdevice) {
    if (nullptr == hdevice)
        return -1;

    vx_device *device = ((vx_device*)hdevice);

#ifdef DUMP_PERF_STATS
    gAutoPerfDump.remove_device(hdevice);
    vx_dump_perf(hdevice, stdout);
#endif

    delete device;

    return 0;
}

extern int vx_dev_caps(vx_device_h hdevice, uint32_t caps_id, uint64_t *value) {
    if (nullptr == hdevice)
        return  -1;

    switch (caps_id) {
    case VX_CAPS_VERSION:
        *value = IMPLEMENTATION_ID;
        break;
    case VX_CAPS_MAX_CORES:
        *value = NUM_CORES * NUM_CLUSTERS;        
        break;
    case VX_CAPS_MAX_WARPS:
        *value = NUM_WARPS;
        break;
    case VX_CAPS_MAX_THREADS:
        *value = NUM_THREADS;
        break;
    case VX_CAPS_CACHE_LINE_SIZE:
        *value = CACHE_BLOCK_SIZE;
        break;
    case VX_CAPS_LOCAL_MEM_SIZE:
        *value = LOCAL_MEM_SIZE;
        break;
    case VX_CAPS_ALLOC_BASE_ADDR:
        *value = ALLOC_BASE_ADDR;
        break;
    case VX_CAPS_KERNEL_BASE_ADDR:
        *value = STARTUP_ADDR;
        break;
    default:
        std::cout << "invalid caps id: " << caps_id << std::endl;
        std::abort();
        return -1;
    }

    return 0;
}

extern int vx_mem_alloc(vx_device_h hdevice, uint64_t size, uint64_t* dev_maddr) {
    if (nullptr == hdevice 
     || nullptr == dev_maddr
     || 0 >= size)
        return -1;

    vx_device *device = ((vx_device*)hdevice);
    return device->alloc_local_mem(size, dev_maddr);
}

extern int vx_mem_free(vx_device_h hdevice, uint64_t dev_maddr) {
    if (nullptr == hdevice)
        return -1;

    vx_device *device = ((vx_device*)hdevice);
    return device->free_local_mem(dev_maddr);
}

extern int vx_buf_alloc(vx_device_h hdevice, uint64_t size, vx_buffer_h* hbuffer) {
    if (nullptr == hdevice 
     || 0 >= size
     || nullptr == hbuffer)
        return -1;

    vx_device *device = ((vx_device*)hdevice);

    auto buffer = new vx_buffer(size, device);
    if (nullptr == buffer->data()) {
        delete buffer;
        return -1;
    }

    *hbuffer = buffer;

    return 0;
}

extern void* vx_host_ptr(vx_buffer_h hbuffer) {
    if (nullptr == hbuffer)
        return nullptr;

    vx_buffer* buffer = ((vx_buffer*)hbuffer);

    return buffer->data();
}

extern int vx_buf_free(vx_buffer_h hbuffer) {
    if (nullptr == hbuffer)
        return -1;

    vx_buffer* buffer = ((vx_buffer*)hbuffer);

    delete buffer;

    return 0;
}

extern int vx_copy_to_dev(vx_buffer_h hbuffer, uint64_t dev_maddr, uint64_t size, uint64_t src_offset) {
    if (nullptr == hbuffer 
     || 0 >= size)
        return -1;

    auto buffer = (vx_buffer*)hbuffer;

    if (size + src_offset > buffer->size())
        return -1;

    return buffer->device()->upload(buffer->data(), dev_maddr, size, src_offset);
}

extern int vx_copy_from_dev(vx_buffer_h hbuffer, uint64_t dev_maddr, uint64_t size, uint64_t dest_offset) {
     if (nullptr == hbuffer 
      || 0 >= size)
        return -1;

    auto buffer = (vx_buffer*)hbuffer;

    if (size + dest_offset > buffer->size())
        return -1;    

    return buffer->device()->download(buffer->data(), dev_maddr, size, dest_offset);
}

extern int vx_start(vx_device_h hdevice) {
    if (nullptr == hdevice)
        return -1;

    vx_device *device = ((vx_device*)hdevice);

    return device->start();
}

extern int vx_ready_wait(vx_device_h hdevice, uint64_t timeout) {
    if (nullptr == hdevice)
        return -1;

    vx_device *device = ((vx_device*)hdevice);

    return device->wait(timeout);
}

