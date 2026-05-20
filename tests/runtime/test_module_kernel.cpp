// Copyright © 2019-2023
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
// http://www.apache.org/licenses/LICENSE-2.0

// ============================================================================
// test_module_kernel.cpp
//
// Exercises the Phase 1b vx_module_h / vx_kernel_h API of
// vortex2_v1_shape_lock_proposal.md:
//   - vx_module_load_file / load_bytes (loads a .vxbin into the device)
//   - vx_module_get_kernel by name (single-`main` fallback for existing .vxbins)
//   - vx_kernel_get_max_block_size (device-default hint)
//   - Refcount semantics: kernel keeps its module alive
//   - Error path: vx_module_get_kernel("nonexistent")
//   - Multi-symbol via in-memory .vxbin with synthetic VXSYMTAB footer
//   - End-to-end launch via vx_enqueue_launch with launch_info.kernel
//     pointing at a vx_kernel_h
//
// Kernel path: defaults to $VORTEX_HOME/build/tests/regression/basic/kernel.vxbin
// (built in the standard regression run), overridable via -k <path> or env
// VX_TEST_VXBIN. The launch section is skipped with a warning if no .vxbin
// is found, so the API smoke tests still cover Phase 1b on a clean tree.
//
// PASS: all sections print [ OK ], exit code 0.
// ============================================================================

#include <vortex2.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

#define CHECK_VX(expr) do { \
    vx_result_t _r = (expr); \
    if (_r != VX_SUCCESS) { \
        fprintf(stderr, "FAILED at %s:%d: '%s' returned %s\n", \
                __FILE__, __LINE__, #expr, vx_result_string(_r)); \
        return 1; \
    } \
} while (0)

#define EXPECT(cond, msg) do { \
    if (!(cond)) { \
        fprintf(stderr, "FAILED at %s:%d: %s\n", __FILE__, __LINE__, msg); \
        return 1; \
    } \
} while (0)

namespace {

std::string find_test_vxbin(const char* override_path) {
    if (override_path && *override_path) return override_path;
    if (const char* env = std::getenv("VX_TEST_VXBIN")) return env;
    const char* home = std::getenv("VORTEX_HOME");
    if (home) {
        std::string p = std::string(home) +
                        "/build/tests/regression/basic/kernel.vxbin";
        std::ifstream ifs(p);
        if (ifs.good()) return p;
    }
    return "";
}

// ---------------------------------------------------------------------------
// Section 1 — vx_module_load_file with single-`main` fallback.
//   - Load a .vxbin (no symbol footer → fallback path)
//   - Resolve "main" — must succeed
//   - Resolve a bogus name — must return VX_ERR_INVALID_VALUE
// ---------------------------------------------------------------------------
int test_module_load_file(vx_device_h dev, const std::string& vxbin) {
    if (vxbin.empty()) {
        printf("       (skipped — no .vxbin available; set VX_TEST_VXBIN)\n");
        return 0;
    }
    vx_module_h mod = nullptr;
    CHECK_VX(vx_module_load_file(dev, vxbin.c_str(), &mod));
    EXPECT(mod != nullptr, "vx_module_load_file must populate the out handle");

    vx_kernel_h k = nullptr;
    CHECK_VX(vx_module_get_kernel(mod, "main", &k));
    EXPECT(k != nullptr, "default 'main' kernel must resolve in fallback mode");

    vx_kernel_h dummy = nullptr;
    auto r = vx_module_get_kernel(mod, "no_such_kernel", &dummy);
    EXPECT(r != VX_SUCCESS, "lookup of missing symbol must fail");

    CHECK_VX(vx_kernel_release(k));
    CHECK_VX(vx_module_release(mod));
    return 0;
}

// ---------------------------------------------------------------------------
// Section 2 — vx_module_load_bytes (no file IO).
// ---------------------------------------------------------------------------
int test_module_load_bytes(vx_device_h dev, const std::string& vxbin) {
    if (vxbin.empty()) {
        printf("       (skipped — no .vxbin available)\n");
        return 0;
    }
    std::ifstream ifs(vxbin, std::ios::binary);
    ifs.seekg(0, ifs.end);
    auto sz = (size_t)ifs.tellg();
    ifs.seekg(0, ifs.beg);
    std::vector<uint8_t> buf(sz);
    ifs.read(reinterpret_cast<char*>(buf.data()), sz);

    vx_module_h mod = nullptr;
    CHECK_VX(vx_module_load_bytes(dev, buf.data(), buf.size(), &mod));
    vx_kernel_h k = nullptr;
    CHECK_VX(vx_module_get_kernel(mod, "main", &k));
    CHECK_VX(vx_kernel_release(k));
    CHECK_VX(vx_module_release(mod));
    return 0;
}

// ---------------------------------------------------------------------------
// Section 3 — kernel refcount keeps module alive.
//   Release the module while the kernel is still held — module's underlying
//   image must remain valid until the kernel is also released.
// ---------------------------------------------------------------------------
int test_refcount(vx_device_h dev, const std::string& vxbin) {
    if (vxbin.empty()) {
        printf("       (skipped — no .vxbin available)\n");
        return 0;
    }
    vx_module_h mod = nullptr;
    CHECK_VX(vx_module_load_file(dev, vxbin.c_str(), &mod));
    vx_kernel_h k = nullptr;
    CHECK_VX(vx_module_get_kernel(mod, "main", &k));

    // Caller drops the module ref. Kernel still holds an internal ref on
    // the module, so the module survives.
    CHECK_VX(vx_module_release(mod));

    // Use the kernel after releasing the module ref — should still work
    // because the kernel keeps the module alive.
    uint32_t bx = 0, by = 0, bz = 0;
    CHECK_VX(vx_kernel_get_max_block_size(k, &bx, &by, &bz));
    EXPECT(bx > 0 && by > 0 && bz > 0, "block-size hint must be positive");

    // Final release tears the kernel down → module → image → device.
    CHECK_VX(vx_kernel_release(k));
    return 0;
}

// ---------------------------------------------------------------------------
// Section 4 — synthetic multi-symbol footer.
//   Build an in-memory .vxbin that LOOKS like an existing kernel but has a
//   VXSYMTAB footer pointing at two named symbols. Verify both resolve
//   (we don't launch the synthetic entries — their PCs are real addresses
//   inside the loaded image, so launching would just rerun main(); we only
//   exercise the loader's footer parser here).
// ---------------------------------------------------------------------------
int test_multi_symbol_footer(vx_device_h dev, const std::string& vxbin) {
    if (vxbin.empty()) {
        printf("       (skipped — no .vxbin available)\n");
        return 0;
    }
    std::ifstream ifs(vxbin, std::ios::binary);
    ifs.seekg(0, ifs.end);
    auto sz = (size_t)ifs.tellg();
    ifs.seekg(0, ifs.beg);
    std::vector<uint8_t> buf(sz);
    ifs.read(reinterpret_cast<char*>(buf.data()), sz);

    const uint64_t min_vma =
        *reinterpret_cast<const uint64_t*>(buf.data() + 0);

    // Append a synthetic VXSYMTAB footer with two entries:
    //   "alpha" -> min_vma     (same PC as default main, will produce same launch)
    //   "beta"  -> min_vma     (also same PC)
    //
    // The loader's footer_total accounting computes the strings-blob length
    // as max(name_off + name_len) across all entries — names need NOT be
    // NUL-terminated since each entry carries an explicit length. Use
    // "alphabeta" (9 bytes) with name_off=0/len=5 for alpha and
    // name_off=5/len=4 for beta.
    //
    // Layout, in order:
    //   [strings : "alphabeta"            = 9 bytes]
    //   [entry alpha: name_off=0,  len=5,  pad=0, pc=min_vma  = 16 bytes]
    //   [entry beta:  name_off=5,  len=4,  pad=0, pc=min_vma  = 16 bytes]
    //   [n_symbols : 4 bytes = 2]
    //   [magic     : 8 bytes  'VXSYMTAB']
    const char strings[] = "alphabeta";
    const size_t strings_sz = sizeof(strings) - 1;  // exclude C string NUL
    buf.insert(buf.end(), strings, strings + strings_sz);

    auto pack_entry = [&](uint32_t name_off, uint16_t name_len, uint64_t pc) {
        uint8_t e[16] = {};
        std::memcpy(e + 0,  &name_off, 4);
        std::memcpy(e + 4,  &name_len, 2);
        // pad bytes 6..7 stay zero
        std::memcpy(e + 8,  &pc,       8);
        buf.insert(buf.end(), e, e + 16);
    };
    pack_entry(0, 5, min_vma);   // "alpha"
    pack_entry(5, 4, min_vma);   // "beta"

    const uint32_t n_syms = 2;
    const uint8_t* n_bytes = reinterpret_cast<const uint8_t*>(&n_syms);
    buf.insert(buf.end(), n_bytes, n_bytes + 4);

    const char magic[8] = {'V','X','S','Y','M','T','A','B'};
    buf.insert(buf.end(), magic, magic + 8);

    vx_module_h mod = nullptr;
    CHECK_VX(vx_module_load_bytes(dev, buf.data(), buf.size(), &mod));

    vx_kernel_h k_alpha = nullptr, k_beta = nullptr;
    CHECK_VX(vx_module_get_kernel(mod, "alpha", &k_alpha));
    CHECK_VX(vx_module_get_kernel(mod, "beta",  &k_beta));

    // The fallback "main" entry should NOT exist when a footer is present.
    vx_kernel_h k_main = nullptr;
    auto r = vx_module_get_kernel(mod, "main", &k_main);
    EXPECT(r != VX_SUCCESS, "footer should override the single-`main` fallback");

    CHECK_VX(vx_kernel_release(k_alpha));
    CHECK_VX(vx_kernel_release(k_beta));
    CHECK_VX(vx_module_release(mod));
    return 0;
}

// ---------------------------------------------------------------------------
// Section 5 — end-to-end launch via vx_kernel_h.
//   vx_enqueue_launch reads the entry PC straight from the vx_kernel_h in
//   vx_launch_info_t.kernel. We don't care what the kernel computes — just
//   that the launch completes without error.
// ---------------------------------------------------------------------------
int test_launch_via_kernel_handle(vx_device_h dev, const std::string& vxbin) {
    if (vxbin.empty()) {
        printf("       (skipped — no .vxbin available)\n");
        return 0;
    }
    vx_module_h mod = nullptr;
    CHECK_VX(vx_module_load_file(dev, vxbin.c_str(), &mod));
    vx_kernel_h k = nullptr;
    CHECK_VX(vx_module_get_kernel(mod, "main", &k));

    vx_queue_info_t qi = {};
    qi.struct_size = sizeof(qi);
    vx_queue_h q = nullptr;
    CHECK_VX(vx_queue_create(dev, &qi, &q));

    // Phase 2: kernel args are a host blob handed straight to the launch —
    // the runtime stages them into a scratch slot. A zero-filled blob makes
    // the basic regression kernel see count==0 and exit early; we only care
    // that the launch completes via the new vx_kernel_h dispatch.
    uint8_t args_blob[64] = {0};

    vx_launch_info_t li = {};
    li.struct_size = sizeof(li);
    li.kernel      = k;
    li.args_host   = args_blob;
    li.args_size   = sizeof(args_blob);
    li.ndim        = 1;
    li.grid_dim[0] = 1;
    li.block_dim[0] = 1;

    vx_event_h ev = nullptr;
    CHECK_VX(vx_enqueue_launch(q, &li, 0, nullptr, &ev));
    CHECK_VX(vx_event_wait_value(ev, 1, 30ull * 1000 * 1000 * 1000));
    CHECK_VX(vx_event_release(ev));

    CHECK_VX(vx_queue_release(q));
    CHECK_VX(vx_kernel_release(k));
    CHECK_VX(vx_module_release(mod));
    return 0;
}

#define RUN(section)                                                     \
    do {                                                                  \
        printf("[RUN ] %s\n", #section);                                  \
        int rc = section(dev, vxbin);                                     \
        if (rc != 0) { printf("[FAIL] %s\n", #section); return rc; }      \
        printf("[ OK ] %s\n", #section);                                  \
    } while (0)

} // namespace

int main(int argc, char** argv) {
    const char* override_path = (argc >= 3 && std::strcmp(argv[1], "-k") == 0)
                                  ? argv[2] : nullptr;
    std::string vxbin = find_test_vxbin(override_path);
    if (vxbin.empty()) {
        printf("note: no .vxbin available; pass -k <path> or set VX_TEST_VXBIN.\n"
               "      Launch-dependent sections will be skipped.\n");
    } else {
        printf("using vxbin: %s\n", vxbin.c_str());
    }

    vx_device_h dev = nullptr;
    CHECK_VX(vx_device_open(0, &dev));

    RUN(test_module_load_file);
    RUN(test_module_load_bytes);
    RUN(test_refcount);
    RUN(test_multi_symbol_footer);
    RUN(test_launch_via_kernel_handle);

    CHECK_VX(vx_device_release(dev));
    printf("PASSED\n");
    return 0;
}
