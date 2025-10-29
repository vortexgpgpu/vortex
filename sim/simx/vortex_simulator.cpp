#include "vortex_simulator.h"
#include <algorithm>
#include <fstream>
#include <limits>
#include <optional>
#include <string>
#include <utility>
#include <vector>
#include "simobject.h"
#include "dcrs.h"
#include <VX_config.h>
#include <VX_types.h>
#include "util.h"

namespace vortex {

VortexSimulator::VortexSimulator()
: arch_(NUM_THREADS, NUM_WARPS, NUM_CORES)
, ram_(0, MEM_PAGE_SIZE)
, proc_(std::make_unique<Processor>(arch_))
// , kernel_image_{}
// , next_alloc_addr_(kAllocBaseAddr)
, halted_(true) {}

bool VortexSimulator::init(const std::string& kernelPath) {
    proc_->attach_ram(&ram_);

    // kernel_image_ = {};
    // next_alloc_addr_ = kAllocBaseAddr;
    // ram_.clear();
    // ram_.set_acl(0, kGlobalMemSize, 0);

    // can be used when launch descriptor is required
    /* bool has_kernel = false;
    if (!kernelPath.empty()) {
        auto image_info = this->loadKernelImage(kernelPath);
        if (!image_info)
            return false;
        kernel_image_ = *image_info;
        has_kernel = true;
    } */

    // Program base DCRs - align startup to loaded kernel when provided
    /* uint64_t startup = STARTUP_ADDR;
    if (has_kernel)
        startup = kernel_image_.base_addr; */

    // setup base DCRs
    const uint64_t startup_addr(STARTUP_ADDR);
    proc_->dcr_write(VX_DCR_BASE_STARTUP_ADDR0, startup_addr & 0xffffffff);
    #if (XLEN == 64)
        proc_->dcr_write(VX_DCR_BASE_STARTUP_ADDR1, startup_addr >> 32);
    #endif
    proc_->dcr_write(VX_DCR_BASE_MPM_CLASS, 0);

    // load program/kernel
    {
      std::string program_ext(fileExtension(kernelPath.c_str()));
      if (program_ext == "bin") {
        std::cout << "vortex_simulator: Loading binary image: " << kernelPath << " with startup address: 0x" << std::hex << startup_addr << std::dec << std::endl;
        ram_.loadBinImage(kernelPath.c_str(), startup_addr);
      } else if (program_ext == "hex") {
        std::cout << "vortex_simulator: Loading hex image: " << kernelPath << std::endl;
        ram_.loadHexImage(kernelPath.c_str());
      } else {
        std::cerr << "Error: only *.bin or *.hex images supported." << std::endl;
        return -1;
      }
    }

    halted_ = false;
    return true;
}

bool VortexSimulator::cycle() {
if (halted_) return false;
//std::cout << "VortexSimulator: cycle()" << std::endl;
// Advance one cycle through the processor interface
bool running = proc_->cycle(); 
halted_ = !running;
//std::cout << "VortexSimulator: cycle() returns " << running << std::endl;
return running;
}

bool VortexSimulator::isHalted() const {
    return halted_;
}

// Required when using launch descriptor and SST memory
/* bool VortexSimulator::allocateMemory(uint64_t size, uint64_t alignment, bool readable, bool writable, uint64_t* addr_out) {
    if (addr_out == nullptr || size == 0)
        return false;

    alignment = normalizeAlignment(alignment);
    uint64_t base = alignUp(next_alloc_addr_, alignment);
    uint64_t end = base + size;
    if (end > kGlobalMemSize)
        return false;

    uint64_t acl_start = alignDown(base, RAM_PAGE_SIZE);
    uint64_t acl_end = alignUp(end, RAM_PAGE_SIZE);
    if (acl_end > kGlobalMemSize)
        return false;

    int flags = 0;
    if (readable) flags |= 0x1;
    if (writable) flags |= 0x2;
    if (flags != 0)
        ram_.set_acl(acl_start, acl_end - acl_start, flags);

    *addr_out = base;
    next_alloc_addr_ = std::max(next_alloc_addr_, acl_end);
    return true;
}

bool VortexSimulator::reserveMemory(uint64_t addr, uint64_t size, bool readable, bool writable) {
    if (size == 0)
        return false;

    uint64_t acl_start = alignDown(addr, RAM_PAGE_SIZE);
    uint64_t acl_end = alignUp(addr + size, RAM_PAGE_SIZE);
    if (acl_end > kGlobalMemSize)
        return false;

    int flags = 0;
    if (readable) flags |= 0x1;
    if (writable) flags |= 0x2;
    ram_.set_acl(acl_start, acl_end - acl_start, flags);

    if (acl_end > next_alloc_addr_)
        next_alloc_addr_ = acl_end;
    return true;
}

void VortexSimulator::setMemoryPermissions(uint64_t addr, uint64_t size, bool readable, bool writable) {
    if (size == 0)
        return;
    uint64_t acl_start = alignDown(addr, RAM_PAGE_SIZE);
    uint64_t acl_end = alignUp(addr + size, RAM_PAGE_SIZE);
    int flags = 0;
    if (readable) flags |= 0x1;
    if (writable) flags |= 0x2;
    ram_.set_acl(acl_start, acl_end - acl_start, flags);
}

void VortexSimulator::writeMemory(uint64_t addr, const void* data, uint64_t size) {
    if (data == nullptr || size == 0)
        return;
    ram_.write(data, addr, size);
}

void VortexSimulator::setStartupArg(uint64_t arg_addr) {
    proc_->dcr_write(VX_DCR_BASE_STARTUP_ARG0, static_cast<uint32_t>(arg_addr & 0xffffffffu));
#if (XLEN == 64)
    proc_->dcr_write(VX_DCR_BASE_STARTUP_ARG1, static_cast<uint32_t>(arg_addr >> 32));
#endif
}

std::optional<KernelImageInfo> VortexSimulator::loadKernelImage(const std::string& path) {
    KernelImageInfo info{};

    if (path.empty())
        return info;

    const auto ext = getFileExt(path);
    if (ext == "bin") {
        std::ifstream ifs(path, std::ios::binary);
        if (!ifs)
            return std::nullopt;

        ifs.seekg(0, std::ios::end);
        const uint64_t size = static_cast<uint64_t>(ifs.tellg());
        ifs.seekg(0, std::ios::beg);
        std::vector<uint8_t> payload(size);
        if (size && !ifs.read(reinterpret_cast<char*>(payload.data()), size))
            return std::nullopt;

        if (!reserveMemory(STARTUP_ADDR, size, true, true))
            return std::nullopt;
        writeMemory(STARTUP_ADDR, payload.data(), size);
        setMemoryPermissions(STARTUP_ADDR, size, true, false);

        info.base_addr = STARTUP_ADDR;
        info.size_bytes = size;
        return info;
    }

    if (ext == "hex") {
        ram_.loadHexImage(path.c_str());
        info.base_addr = STARTUP_ADDR;
        info.size_bytes = 0;
        return info;
    }

    if (ext == "vxbin") {
        std::ifstream ifs(path, std::ios::binary);
        if (!ifs)
            return std::nullopt;

        uint64_t min_vma = 0;
        uint64_t max_vma = 0;

        ifs.read(reinterpret_cast<char*>(&min_vma), sizeof(uint64_t));
        ifs.read(reinterpret_cast<char*>(&max_vma), sizeof(uint64_t));
        if (!ifs || max_vma < min_vma)
            return std::nullopt;

        constexpr size_t header_bytes = sizeof(uint64_t) * 2;
        ifs.seekg(0, std::ios::end);
        const size_t file_size = static_cast<size_t>(ifs.tellg());
        if (file_size < header_bytes)
            return std::nullopt;

        const uint64_t payload_size = static_cast<uint64_t>(file_size - header_bytes);
        const uint64_t image_span   = max_vma - min_vma;
        if (image_span == 0)
            return std::nullopt;
        ifs.seekg(header_bytes, std::ios::beg);

        std::vector<uint8_t> payload(payload_size);
        if (payload_size && !ifs.read(reinterpret_cast<char*>(payload.data()), payload_size))
            return std::nullopt;

        if (!reserveMemory(min_vma, image_span, true, true))
            return std::nullopt;
        if (payload_size)
            writeMemory(min_vma, payload.data(), payload_size);
        if (image_span > payload_size) {
            std::vector<uint8_t> zeros(static_cast<size_t>(image_span - payload_size), 0);
            writeMemory(min_vma + payload_size, zeros.data(), zeros.size());
        }
        setMemoryPermissions(min_vma, image_span, true, false);

        info.base_addr = min_vma;
        info.size_bytes = image_span;
        return info;
    }

    return std::nullopt;
}

uint64_t VortexSimulator::alignUp(uint64_t value, uint64_t alignment) {
    return (value + alignment - 1) & ~(alignment - 1);
}

uint64_t VortexSimulator::alignDown(uint64_t value, uint64_t alignment) {
    return value & ~(alignment - 1);
}

uint64_t VortexSimulator::normalizeAlignment(uint64_t alignment) {
    if (alignment == 0)
        alignment = kDefaultAlignment;
    if (alignment < kDefaultAlignment)
        alignment = kDefaultAlignment;
    if ((alignment & (alignment - 1)) == 0)
        return alignment;

    alignment--;
    alignment |= alignment >> 1;
    alignment |= alignment >> 2;
    alignment |= alignment >> 4;
    alignment |= alignment >> 8;
    alignment |= alignment >> 16;
    alignment |= alignment >> 32;
    alignment++;
    return alignment;
} */

} // namespace vortex
