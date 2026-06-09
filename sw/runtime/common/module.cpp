// Copyright © 2019-2023
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
// http://www.apache.org/licenses/LICENSE-2.0

// ============================================================================
// module.cpp
//
// Module = a loaded .vxbin (device-side image + parsed symbol table).
// Kernel = a named entry point inside a module, with its PC cached.
//
// .vxbin layout:
//   [min_vma   : 8 bytes LE]
//   [max_vma   : 8 bytes LE]
//   [binary bytes ...]                                     <- length = bin_sz
//   --- optional symbol-table footer ---
//   [string blob (variable, all symbol names back-to-back, NUL-separated)]
//   [entries: N × { name_off:4, name_len:2, _pad:2, pc:8 }  = 16 bytes each ]
//   [n_symbols : 4 bytes LE]
//   [magic     : 8 bytes 'VXSYMTAB']                       <- end of file
//
// Loader checks the last 8 bytes for the magic. If present, parses
// backward to recover the symbol table. If absent, falls back to a single
// "main" entry whose PC is min_vma.
// ============================================================================

#include "vortex2_internal.h"

#include <cstdint>
#include <cstring>
#include <fstream>
#include <vector>

namespace vx {

// ----------------------------------------------------------------------------
// Module
// ----------------------------------------------------------------------------

Module::Module(Device* dev, Buffer* image, uint64_t base_addr)
    : device_(dev), image_(image), base_addr_(base_addr) {
    device_->retain();
    // image_ ref was already created by Buffer::reserve.
}

Module::~Module() {
    // Cached kernels hold a ref on this module; once we're being deleted,
    // they're already gone (release order ensures kernels drop us first).
    if (image_) image_->release();
    if (device_) device_->release();
}

vx_result_t Module::load_file(Device* dev, const char* path, Module** out) {
    if (!dev || !path || !out) return VX_ERR_INVALID_VALUE;
    std::ifstream ifs(path, std::ios::binary);
    if (!ifs) return VX_ERR_INVALID_VALUE;
    ifs.seekg(0, ifs.end);
    auto file_sz = (size_t)ifs.tellg();
    ifs.seekg(0, ifs.beg);
    if (file_sz < 16) return VX_ERR_INVALID_VALUE;

    std::vector<uint8_t> all(file_sz);
    ifs.read(reinterpret_cast<char*>(all.data()), file_sz);
    if (!ifs) return VX_ERR_INVALID_VALUE;
    return load_bytes(dev, all.data(), all.size(), out);
}

vx_result_t Module::load_bytes(Device* dev, const void* bytes_, size_t size,
                               Module** out) {
    if (!dev || !bytes_ || size < 16 || !out) return VX_ERR_INVALID_VALUE;
    auto* bytes = static_cast<const uint8_t*>(bytes_);

    const uint64_t min_vma = *reinterpret_cast<const uint64_t*>(bytes + 0);
    const uint64_t max_vma = *reinterpret_cast<const uint64_t*>(bytes + 8);
    if (max_vma <= min_vma) return VX_ERR_INVALID_VALUE;
    const uint64_t rt_sz = max_vma - min_vma;

    // Sniff for the symbol footer. It lives at the END of the file; the bin
    // proper is everything between the 16-byte header and (start-of-footer).
    static const char kFooterMagic[8] = {'V','X','S','Y','M','T','A','B'};
    size_t footer_total = 0;
    uint32_t n_syms = 0;
    bool has_footer = false;
    if (size >= 16 + 12 &&
        std::memcmp(bytes + size - 8, kFooterMagic, 8) == 0) {
        n_syms = *reinterpret_cast<const uint32_t*>(bytes + size - 12);
        const size_t entries_sz = size_t(n_syms) * 16;
        if (size >= 16 + 12 + entries_sz) {
            footer_total = 12 + entries_sz;
            // String blob sits between the bin and the entries; we figure
            // its size out from the entries' (name_off, name_len) pairs.
            const uint8_t* entries_start = bytes + size - 12 - entries_sz;
            size_t max_end_off = 0;
            for (uint32_t i = 0; i < n_syms; ++i) {
                const uint8_t* e = entries_start + size_t(i) * 16;
                const uint32_t name_off = *reinterpret_cast<const uint32_t*>(e + 0);
                const uint16_t name_len = *reinterpret_cast<const uint16_t*>(e + 4);
                const size_t end_off = size_t(name_off) + name_len;
                if (end_off > max_end_off) max_end_off = end_off;
            }
            footer_total += max_end_off;   // string blob included
            if (footer_total <= size - 16) {
                has_footer = true;
            } else {
                footer_total = 0;
                n_syms = 0;
            }
        }
    }

    const uint64_t bin_sz = size - 16 - footer_total;
    if (bin_sz > rt_sz) return VX_ERR_INVALID_VALUE;
    const uint8_t* bin = bytes + 16;

    // Reserve the image's VMA range on the device.
    Buffer* image = nullptr;
    auto r = Buffer::reserve(dev, min_vma, rt_sz, 0, &image);
    if (r != VX_SUCCESS) return r;

    // .text/.rodata read-only, .bss read-write.
    r = image->access(0, bin_sz, VX_MEM_READ);
    if (r != VX_SUCCESS) { image->release(); return r; }
    if (rt_sz > bin_sz) {
        r = image->access(bin_sz, rt_sz - bin_sz, VX_MEM_READ_WRITE);
        if (r != VX_SUCCESS) { image->release(); return r; }
    }

    // Synchronously upload the binary payload + zero the BSS region.
    // Routed via dev_write (the CP's DMA on a CP-only-DMA backend); still
    // synchronous, so Module stays a pure synchronous primitive — the
    // caller doesn't need a queue handy to load a module.
    r = dev->dev_write(min_vma, bin, bin_sz);
    if (r != VX_SUCCESS) { image->release(); return r; }
    if (rt_sz > bin_sz) {
        std::vector<uint8_t> zeros(rt_sz - bin_sz, 0);
        r = dev->dev_write(min_vma + bin_sz, zeros.data(), rt_sz - bin_sz);
        if (r != VX_SUCCESS) { image->release(); return r; }
    }

    // Build the Module. Symbol table comes from the footer if present;
    // otherwise we fall back to a single "main" entry at min_vma.
    Module* m = new Module(dev, image, min_vma);
    if (has_footer) {
        const uint8_t* entries_start = bytes + size - 12 - size_t(n_syms) * 16;
        const uint8_t* strings_start = bytes + 16 + bin_sz;
        for (uint32_t i = 0; i < n_syms; ++i) {
            const uint8_t* e = entries_start + size_t(i) * 16;
            const uint32_t name_off = *reinterpret_cast<const uint32_t*>(e + 0);
            const uint16_t name_len = *reinterpret_cast<const uint16_t*>(e + 4);
            const uint64_t pc       = *reinterpret_cast<const uint64_t*>(e + 8);
            m->symbols_.push_back({
                std::string(reinterpret_cast<const char*>(strings_start + name_off),
                            name_len),
                pc
            });
        }
        // A single-kernel vxbin (the regression tests) carries exactly one
        // entry stub. Expose "main" @ min_vma too — the same launch entry
        // used for stub-less single-entry images — so callers using the
        // conventional entry name resolve it whether or not the compiler
        // emitted a per-kernel entry stub. Multi-kernel images keep their
        // explicit names.
        if (m->symbols_.size() == 1 && m->symbols_[0].name != "main") {
            m->symbols_.push_back({"main", min_vma});
        }
    } else {
        m->symbols_.push_back({"main", min_vma});
    }

    *out = m;
    return VX_SUCCESS;
}

vx_result_t Module::get_kernel(const char* name, Kernel** out) {
    if (!name || !out) return VX_ERR_INVALID_VALUE;
    std::lock_guard<std::mutex> g(kcache_mu_);
    auto it = kernel_cache_.find(name);
    if (it != kernel_cache_.end()) {
        it->second->retain();
        *out = it->second;
        return VX_SUCCESS;
    }
    for (const auto& s : symbols_) {
        if (s.name == name) {
            Kernel* k = nullptr;
            auto r = Kernel::create(this, s.pc, &k);
            if (r != VX_SUCCESS) return r;
            // Cache holds a non-owning reference — Kernel's destructor
            // removes itself from the cache.
            kernel_cache_[name] = k;
            // The kernel was created with refcount=1; return that ref to
            // the caller. The cache's "reference" is a raw pointer only.
            *out = k;
            return VX_SUCCESS;
        }
    }
    return VX_ERR_INVALID_VALUE;
}

// ----------------------------------------------------------------------------
// Kernel
// ----------------------------------------------------------------------------

Kernel::Kernel(Module* mod, uint64_t pc)
    : module_(mod), pc_(pc) {
    module_->retain();
}

Kernel::~Kernel() {
    // Best-effort removal from the module's cache (in case some lookup
    // raced; the cache holds a weak ref so it's safe to leave a dangling
    // entry briefly, but cleaning up keeps the map size bounded over
    // many create/release cycles of the same kernel name).
    if (module_) {
        std::lock_guard<std::mutex> g(module_->kcache_mu_);
        for (auto it = module_->kernel_cache_.begin();
             it != module_->kernel_cache_.end(); ++it) {
            if (it->second == this) {
                module_->kernel_cache_.erase(it);
                break;
            }
        }
        module_->release();
    }
}

vx_result_t Kernel::create(Module* mod, uint64_t pc, Kernel** out) {
    if (!mod || !out) return VX_ERR_INVALID_VALUE;
    *out = new Kernel(mod, pc);
    return VX_SUCCESS;
}

vx_result_t Kernel::get_max_block_size(uint32_t* x, uint32_t* y, uint32_t* z) {
    if (!x || !y || !z) return VX_ERR_INVALID_VALUE;
    // Default block size: full warp width × num_warps × 1.
    uint64_t nt = 0, nw = 0;
    auto* dev = module_->device();
    auto r = dev->query_caps(VX_CAPS_NUM_THREADS, &nt);
    if (r != VX_SUCCESS) return r;
    r = dev->query_caps(VX_CAPS_NUM_WARPS, &nw);
    if (r != VX_SUCCESS) return r;
    *x = (uint32_t)nt;
    *y = (uint32_t)nw;
    *z = 1;
    return VX_SUCCESS;
}

} // namespace vx

// ============================================================================
// C entry points
// ============================================================================

using namespace vx;

extern "C" vx_result_t vx_module_load_file(vx_device_h dev, const char* path,
                                           vx_module_h* out) {
    VX_C_ENTRY_TRY
    if (!dev || !path || !out) return VX_ERR_INVALID_VALUE;
    Module* m = nullptr;
    auto r = Module::load_file(to_device(dev), path, &m);
    if (r != VX_SUCCESS) return r;
    *out = to_handle(m);
    return VX_SUCCESS;
    VX_C_ENTRY_CATCH
}

extern "C" vx_result_t vx_module_load_bytes(vx_device_h dev, const void* bytes,
                                            size_t size, vx_module_h* out) {
    VX_C_ENTRY_TRY
    if (!dev || !bytes || !out) return VX_ERR_INVALID_VALUE;
    Module* m = nullptr;
    auto r = Module::load_bytes(to_device(dev), bytes, size, &m);
    if (r != VX_SUCCESS) return r;
    *out = to_handle(m);
    return VX_SUCCESS;
    VX_C_ENTRY_CATCH
}

extern "C" vx_result_t vx_module_retain(vx_module_h mod) {
    if (!mod) return VX_ERR_INVALID_HANDLE;
    to_module(mod)->retain();
    return VX_SUCCESS;
}

extern "C" vx_result_t vx_module_release(vx_module_h mod) {
    if (!mod) return VX_ERR_INVALID_HANDLE;
    to_module(mod)->release();
    return VX_SUCCESS;
}

extern "C" vx_result_t vx_module_get_kernel(vx_module_h mod, const char* name,
                                            vx_kernel_h* out) {
    VX_C_ENTRY_TRY
    if (!mod || !name || !out) return VX_ERR_INVALID_VALUE;
    Kernel* k = nullptr;
    auto r = to_module(mod)->get_kernel(name, &k);
    if (r != VX_SUCCESS) return r;
    *out = to_handle(k);
    return VX_SUCCESS;
    VX_C_ENTRY_CATCH
}

extern "C" vx_result_t vx_kernel_retain(vx_kernel_h k) {
    if (!k) return VX_ERR_INVALID_HANDLE;
    to_kernel(k)->retain();
    return VX_SUCCESS;
}

extern "C" vx_result_t vx_kernel_release(vx_kernel_h k) {
    if (!k) return VX_ERR_INVALID_HANDLE;
    to_kernel(k)->release();
    return VX_SUCCESS;
}

extern "C" vx_result_t vx_kernel_get_max_block_size(vx_kernel_h k,
                                                    uint32_t* x, uint32_t* y,
                                                    uint32_t* z) {
    if (!k) return VX_ERR_INVALID_HANDLE;
    return to_kernel(k)->get_max_block_size(x, y, z);
}
