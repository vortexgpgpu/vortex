// Copyright © 2019-2023
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
// http://www.apache.org/licenses/LICENSE-2.0

// ============================================================================
// test_indirect.cpp
//
// Exercises vx_launch_indirect_t (command-processor indirect launch):
//   - VX_CAPS_CP_INDIRECT
//   - a grid written by a kernel and consumed by the next launch, 1D and 3D
//   - a zero in each grid dimension skips the launch and its skip range
//   - predicate true / false / inverted, and a predicate-only command
//   - a skip range spanning launches and DCR writes, in a ring batch and in a
//     draw bundle
//   - a draw bundle of several hundred launches, run to its last command
//   - an indirect-count unroll with the count below, at and above the bound
//   - byte-count derivation (bias, odd divisor, ceil-shift)
//   - a record whose words straddle a cache line
//   - vx_enqueue_launch with a chained struct
//   - validation failures
//
// Kernels come from tests/regression/indirect/kernel.vxbin (VX_TEST_INDIRECT_VXBIN
// or -k <path>).
//
// PASS: all sections print [ OK ], exit code 0.
// ============================================================================

#include <vortex2.h>
#include <VX_types.h>

#include "../regression/indirect/common.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
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

#define EXPECT(cond, ...) do { \
    if (!(cond)) { \
        fprintf(stderr, "FAILED at %s:%d: ", __FILE__, __LINE__); \
        fprintf(stderr, __VA_ARGS__); \
        fprintf(stderr, "\n"); \
        return 1; \
    } \
} while (0)

namespace {

constexpr uint32_t HITS_WORDS = 256;
constexpr uint32_t DIMS_WORDS = 16;

vx_device_h g_dev = nullptr;
vx_queue_h  g_q   = nullptr;
vx_kernel_h g_produce = nullptr;
vx_kernel_h g_mark    = nullptr;
uint32_t    g_nt      = 1;

// One mark_k target: hits[linear block] and dims {grid.xyz, launches}.
struct Mark {
    vx_buffer_h hits = nullptr;
    vx_buffer_h dims = nullptr;
    mark_arg_t  arg  = {};
    std::vector<uint32_t> h_hits = std::vector<uint32_t>(HITS_WORDS);
    std::vector<uint32_t> h_dims = std::vector<uint32_t>(DIMS_WORDS);
};

int mark_create(Mark& m) {
    CHECK_VX(vx_buffer_create(g_dev, HITS_WORDS * 4, VX_MEM_READ_WRITE, &m.hits));
    CHECK_VX(vx_buffer_create(g_dev, DIMS_WORDS * 4, VX_MEM_READ_WRITE, &m.dims));
    CHECK_VX(vx_buffer_address(m.hits, &m.arg.hits_addr));
    CHECK_VX(vx_buffer_address(m.dims, &m.arg.dims_addr));
    return 0;
}

int mark_clear(Mark& m) {
    std::vector<uint32_t> z(HITS_WORDS, 0);
    CHECK_VX(vx_enqueue_write(g_q, m.hits, 0, z.data(), HITS_WORDS * 4, 0, nullptr, nullptr));
    CHECK_VX(vx_enqueue_write(g_q, m.dims, 0, z.data(), DIMS_WORDS * 4, 0, nullptr, nullptr));
    CHECK_VX(vx_queue_finish(g_q, VX_TIMEOUT_INFINITE));
    return 0;
}

int mark_read(Mark& m) {
    CHECK_VX(vx_enqueue_read(g_q, m.h_hits.data(), m.hits, 0, HITS_WORDS * 4, 0, nullptr, nullptr));
    CHECK_VX(vx_enqueue_read(g_q, m.h_dims.data(), m.dims, 0, DIMS_WORDS * 4, 0, nullptr, nullptr));
    CHECK_VX(vx_queue_finish(g_q, VX_TIMEOUT_INFINITE));
    return 0;
}

void mark_release(Mark& m) {
    vx_buffer_release(m.hits);
    vx_buffer_release(m.dims);
}

// Expect exactly `launches` launches of a gx*gy*gz grid (0 launches: untouched).
int mark_expect(Mark& m, uint32_t launches, uint32_t gx, uint32_t gy, uint32_t gz,
                const char* what) {
    if (mark_read(m)) {
        return 1;
    }
    EXPECT(m.h_dims[3] == launches, "%s: %u launches, expected %u", what, m.h_dims[3], launches);
    const uint32_t blocks = launches ? gx * gy * gz : 0;
    if (launches) {
        EXPECT(m.h_dims[0] == gx && m.h_dims[1] == gy && m.h_dims[2] == gz,
               "%s: grid %ux%ux%u, expected %ux%ux%u", what,
               m.h_dims[0], m.h_dims[1], m.h_dims[2], gx, gy, gz);
    }
    for (uint32_t b = 0; b < HITS_WORDS; ++b) {
        const uint32_t want = (b < blocks) ? launches : 0;
        EXPECT(m.h_hits[b] == want, "%s: block %u ran %u times, expected %u",
               what, b, m.h_hits[b], want);
    }
    return 0;
}

vx_launch_info_t mark_launch(const Mark& m, uint32_t ndim, uint32_t gx, uint32_t gy,
                             uint32_t gz) {
    vx_launch_info_t li = {};
    li.struct_size  = sizeof(li);
    li.kernel       = g_mark;
    li.args_host    = &m.arg;
    li.args_size    = sizeof(m.arg);
    li.ndim         = ndim;
    li.grid_dim[0]  = gx;
    li.grid_dim[1]  = gy;
    li.grid_dim[2]  = gz;
    li.block_dim[0] = g_nt;
    li.block_dim[1] = 1;
    li.block_dim[2] = 1;
    return li;
}

vx_launch_indirect_t indirect() {
    vx_launch_indirect_t x = {};
    x.struct_size = sizeof(x);
    x.type        = VX_STRUCT_TYPE_LAUNCH_INDIRECT;
    x.grid_word[0] = 0;
    x.grid_word[1] = 1;
    x.grid_word[2] = 2;
    return x;
}

vx_command_t launch_cmd(const vx_launch_info_t* li) {
    vx_command_t c = {};
    c.type = VX_COMMAND_LAUNCH;
    c.data.launch = li;
    return c;
}

vx_command_t dcr_cmd(uint32_t addr, uint32_t value) {
    vx_command_t c = {};
    c.type = VX_COMMAND_DCR_WRITE;
    c.data.dcr.addr  = addr;
    c.data.dcr.value = value;
    return c;
}

vx_result_t submit(const std::vector<vx_command_t>& cmds, bool bundle = false) {
    vx_result_t r = bundle
        ? vx_enqueue_draw(g_q, cmds.data(), uint32_t(cmds.size()), 0, nullptr, nullptr)
        : vx_enqueue_commands(g_q, cmds.data(), uint32_t(cmds.size()), 0, nullptr, nullptr);
    if (r != VX_SUCCESS) {
        return r;
    }
    return vx_queue_finish(g_q, VX_TIMEOUT_INFINITE);
}

// Device scratch for records / predicates, written by the host.
struct Words {
    vx_buffer_h buf  = nullptr;
    uint64_t    addr = 0;
};

int words_create(Words& w, uint32_t bytes) {
    CHECK_VX(vx_buffer_create(g_dev, bytes, VX_MEM_READ_WRITE, &w.buf));
    CHECK_VX(vx_buffer_address(w.buf, &w.addr));
    return 0;
}

int words_write(Words& w, uint32_t off, const std::vector<uint32_t>& v) {
    CHECK_VX(vx_enqueue_write(g_q, w.buf, off, v.data(), v.size() * 4, 0, nullptr, nullptr));
    CHECK_VX(vx_queue_finish(g_q, VX_TIMEOUT_INFINITE));
    return 0;
}

// ---------------------------------------------------------------------------

int test_caps() {
    uint64_t v = 0;
    CHECK_VX(vx_device_query(g_dev, VX_CAPS_CP_INDIRECT, &v));
    EXPECT(v == 1, "VX_CAPS_CP_INDIRECT = %llu, expected 1", (unsigned long long)v);
    return 0;
}

// A kernel writes the grid; the next launch in the same batch consumes it.
int test_device_grid() {
    Words rec;
    if (words_create(rec, 64)) {
        return 1;
    }
    if (words_write(rec, 0, std::vector<uint32_t>(16, 0))) {
        return 1;
    }
    Mark m3, m1;
    if (mark_create(m3) || mark_create(m1) || mark_clear(m3) || mark_clear(m1)) {
        return 1;
    }
    produce_arg_t pa = {};
    pa.dst_addr  = rec.addr;
    pa.num_words = 4;
    pa.words[0]  = 3;
    pa.words[1]  = 5;
    pa.words[2]  = 2;
    pa.words[3]  = 37;
    vx_launch_info_t lp = {};
    lp.struct_size  = sizeof(lp);
    lp.kernel       = g_produce;
    lp.args_host    = &pa;
    lp.args_size    = sizeof(pa);
    lp.ndim         = 1;
    lp.grid_dim[0]  = 1;
    lp.block_dim[0] = 1;

    vx_launch_info_t l3 = mark_launch(m3, 3, 0, 0, 0);
    vx_launch_indirect_t x3 = indirect();
    x3.record_addr = rec.addr;
    l3.next = &x3;

    // 1D, word 3 (37) at shift 2 -> ceil(37 / 4) = 10 blocks.
    vx_launch_info_t l1 = mark_launch(m1, 1, 0, 0, 0);
    vx_launch_indirect_t x1 = indirect();
    x1.record_addr   = rec.addr;
    x1.grid_word[0]  = 3;
    x1.grid_shift[0] = 2;
    l1.next = &x1;

    CHECK_VX(submit({launch_cmd(&lp), launch_cmd(&l3), launch_cmd(&l1)}));
    if (mark_expect(m3, 1, 3, 5, 2, "device grid 3D") ||
        mark_expect(m1, 1, 10, 1, 1, "device grid 1D shift")) {
        return 1;
    }
    mark_release(m3);
    mark_release(m1);
    vx_buffer_release(rec.buf);
    return 0;
}

// A zero dimension skips the launch and its skip range, never a later command.
int test_zero_dims() {
    Words rec;
    if (words_create(rec, 64)) {
        return 1;
    }
    for (int d = 0; d < 3; ++d) {
        std::vector<uint32_t> w = {2, 2, 2};
        w[d] = 0;
        if (words_write(rec, 0, w)) {
            return 1;
        }
        Mark mz, ms, mr;
        if (mark_create(mz) || mark_create(ms) || mark_create(mr) ||
            mark_clear(mz) || mark_clear(ms) || mark_clear(mr)) {
            return 1;
        }
        vx_launch_info_t lz = mark_launch(mz, 3, 0, 0, 0);
        vx_launch_indirect_t xz = indirect();
        xz.record_addr = rec.addr;
        xz.skip_count  = 1;
        lz.next = &xz;
        vx_launch_info_t ls = mark_launch(ms, 1, 1, 1, 1);
        vx_launch_info_t lr = mark_launch(mr, 1, 2, 1, 1);
        CHECK_VX(submit({launch_cmd(&lz), launch_cmd(&ls), launch_cmd(&lr)}));
        if (mark_expect(mz, 0, 0, 0, 0, "zero-dim launch") ||
            mark_expect(ms, 0, 0, 0, 0, "zero-dim skip range") ||
            mark_expect(mr, 1, 2, 1, 1, "after zero-dim skip range")) {
            fprintf(stderr, "  (zero in dimension %d)\n", d);
            return 1;
        }
        mark_release(mz);
        mark_release(ms);
        mark_release(mr);
    }
    vx_buffer_release(rec.buf);
    return 0;
}

// exec = (*pred > ref) ^ invert, unsigned.
int test_predicate() {
    Words pred;
    if (words_create(pred, 64) || words_write(pred, 0, {5, 0xFFFFFFFFu})) {
        return 1;
    }
    struct Case { uint32_t off, ref; bool inv, runs; };
    const Case cases[] = {
        {0, 4, false, true},  {0, 5, false, false},
        {0, 5, true,  true},  {0, 4, true,  false},
        {4, 0xFFFFFFFEu, false, true}, {4, 0xFFFFFFFFu, false, false},
    };
    for (const Case& c : cases) {
        Mark m;
        if (mark_create(m) || mark_clear(m)) {
            return 1;
        }
        vx_launch_info_t l = mark_launch(m, 2, 2, 3, 1);
        vx_launch_indirect_t x = indirect();
        x.pred_addr = pred.addr + c.off;
        x.pred_ref  = c.ref;
        x.flags     = c.inv ? VX_INDIRECT_PRED_INVERT : 0;
        l.next = &x;
        CHECK_VX(submit({launch_cmd(&l)}));
        if (mark_expect(m, c.runs ? 1 : 0, 2, 3, 1, "predicate")) {
            fprintf(stderr, "  (pred word at +%u, ref=0x%x, invert=%d)\n", c.off, c.ref, c.inv);
            return 1;
        }
        mark_release(m);
    }
    vx_buffer_release(pred.buf);
    return 0;
}

// A skip range of launch, DCR write, launch; the command after it runs.
int test_skip_range(bool bundle) {
    Words pred;
    if (words_create(pred, 64)) {
        return 1;
    }
    for (uint32_t p = 0; p < 2; ++p) {
        if (words_write(pred, 0, {p})) {
            return 1;
        }
        Mark ma, mb, mc, md;
        if (mark_create(ma) || mark_create(mb) || mark_create(mc) || mark_create(md) ||
            mark_clear(ma) || mark_clear(mb) || mark_clear(mc) || mark_clear(md)) {
            return 1;
        }
        vx_launch_info_t la = mark_launch(ma, 1, 2, 1, 1);
        vx_launch_indirect_t xa = indirect();
        xa.pred_addr  = pred.addr;
        xa.skip_count = 3;
        la.next = &xa;
        vx_launch_info_t lb = mark_launch(mb, 1, 3, 1, 1);
        vx_launch_info_t lc = mark_launch(mc, 1, 4, 1, 1);
        vx_launch_info_t ld = mark_launch(md, 1, 5, 1, 1);
        CHECK_VX(submit({launch_cmd(&la), launch_cmd(&lb),
                         dcr_cmd(VX_DCR_KMU_LMEM_SIZE, 0),
                         launch_cmd(&lc), launch_cmd(&ld)}, bundle));
        const uint32_t n = p;
        if (mark_expect(ma, n, 2, 1, 1, "skip head") ||
            mark_expect(mb, n, 3, 1, 1, "skip range launch 1") ||
            mark_expect(mc, n, 4, 1, 1, "skip range launch 2") ||
            mark_expect(md, 1, 5, 1, 1, "after skip range")) {
            fprintf(stderr, "  (pred=%u, %s)\n", p, bundle ? "draw bundle" : "ring batch");
            return 1;
        }
        mark_release(ma);
        mark_release(mb);
        mark_release(mc);
        mark_release(md);
    }
    vx_buffer_release(pred.buf);
    return 0;
}

// A skipped launch whose range ends on a predicate-only guard. The guard is
// skipped too, so it cannot skip what follows it. A range that is short by the
// skipped launch's own trailing flush lands on the guard instead, which then
// skips the last launch.
int test_skip_into_guard(bool bundle) {
    Words pred;
    if (words_create(pred, 64) || words_write(pred, 0, {0})) {
        return 1;
    }
    Mark ml, mm;
    if (mark_create(ml) || mark_create(mm) || mark_clear(ml) || mark_clear(mm)) {
        return 1;
    }
    vx_launch_info_t ll = mark_launch(ml, 1, 2, 1, 1);
    vx_launch_indirect_t xl = indirect();
    xl.pred_addr  = pred.addr;
    xl.skip_count = 1;
    ll.next = &xl;
    vx_launch_info_t lg = {};
    lg.struct_size = sizeof(lg);
    vx_launch_indirect_t xg = indirect();
    xg.pred_addr  = pred.addr;
    xg.flags      = VX_INDIRECT_NO_LAUNCH;
    xg.skip_count = 1;
    lg.next = &xg;
    vx_launch_info_t lm = mark_launch(mm, 1, 3, 1, 1);
    CHECK_VX(submit({launch_cmd(&ll), launch_cmd(&lg), launch_cmd(&lm)}, bundle));
    if (mark_expect(ml, 0, 2, 1, 1, "skipped launch") ||
        mark_expect(mm, 1, 3, 1, 1, "launch after the skipped guard")) {
        fprintf(stderr, "  (%s)\n", bundle ? "draw bundle" : "ring batch");
        return 1;
    }
    mark_release(ml);
    mark_release(mm);
    vx_buffer_release(pred.buf);
    return 0;
}

// A predicate-only command guards the two launches behind it.
int test_no_launch(bool bundle) {
    Words pred;
    if (words_create(pred, 64)) {
        return 1;
    }
    for (uint32_t p = 0; p < 2; ++p) {
        if (words_write(pred, 0, {p})) {
            return 1;
        }
        Mark ma, mb, mc;
        if (mark_create(ma) || mark_create(mb) || mark_create(mc) ||
            mark_clear(ma) || mark_clear(mb) || mark_clear(mc)) {
            return 1;
        }
        vx_launch_info_t lg = {};
        lg.struct_size = sizeof(lg);
        vx_launch_indirect_t xg = indirect();
        xg.pred_addr  = pred.addr;
        xg.flags      = VX_INDIRECT_NO_LAUNCH;
        xg.skip_count = 2;
        lg.next = &xg;
        vx_launch_info_t la = mark_launch(ma, 1, 2, 1, 1);
        vx_launch_info_t lb = mark_launch(mb, 1, 3, 1, 1);
        vx_launch_info_t lc = mark_launch(mc, 1, 1, 1, 1);
        CHECK_VX(submit({launch_cmd(&lg), launch_cmd(&la), launch_cmd(&lb),
                         launch_cmd(&lc)}, bundle));
        if (mark_expect(ma, p, 2, 1, 1, "guarded launch 1") ||
            mark_expect(mb, p, 3, 1, 1, "guarded launch 2") ||
            mark_expect(mc, 1, 1, 1, 1, "after guard")) {
            fprintf(stderr, "  (pred=%u, %s)\n", p, bundle ? "draw bundle" : "ring batch");
            return 1;
        }
        mark_release(ma);
        mark_release(mb);
        mark_release(mc);
    }
    vx_buffer_release(pred.buf);
    return 0;
}

// A draw bundle runs every command it holds, however many: the same commands
// as a ring batch, which the command processor streams to the end. Each launch
// is two of the bundle's steps, a launch and a flush.
int test_long_bundle() {
    constexpr uint32_t N = 300;
    Mark ma, mb;
    if (mark_create(ma) || mark_create(mb) || mark_clear(ma) || mark_clear(mb)) {
        return 1;
    }
    vx_launch_info_t la = mark_launch(ma, 1, 1, 1, 1);
    vx_launch_info_t lb = mark_launch(mb, 1, 2, 1, 1);
    std::vector<vx_command_t> cmds(N, launch_cmd(&la));
    cmds.push_back(launch_cmd(&lb));
    CHECK_VX(submit(cmds, true));
    if (mark_expect(ma, N, 1, 1, 1, "every launch of a long bundle") ||
        mark_expect(mb, 1, 2, 1, 1, "the bundle's last launch")) {
        return 1;
    }
    mark_release(ma);
    mark_release(mb);
    return 0;
}

// Indirect count with a recording bound: launch d runs iff count > d.
int test_count_unroll() {
    constexpr uint32_t MAX = 4;
    Words cnt;
    if (words_create(cnt, 64)) {
        return 1;
    }
    const uint32_t counts[] = {2, MAX, MAX + 2, 0};
    for (uint32_t count : counts) {
        if (words_write(cnt, 0, {count})) {
            return 1;
        }
        Mark m[MAX];
        vx_launch_info_t l[MAX];
        vx_launch_indirect_t x[MAX];
        std::vector<vx_command_t> cmds;
        for (uint32_t d = 0; d < MAX; ++d) {
            if (mark_create(m[d]) || mark_clear(m[d])) {
                return 1;
            }
            l[d] = mark_launch(m[d], 1, d + 1, 1, 1);
            x[d] = indirect();
            x[d].pred_addr = cnt.addr;
            x[d].pred_ref  = d;
            l[d].next = &x[d];
            cmds.push_back(launch_cmd(&l[d]));
        }
        CHECK_VX(submit(cmds));
        for (uint32_t d = 0; d < MAX; ++d) {
            if (mark_expect(m[d], (count > d) ? 1 : 0, d + 1, 1, 1, "count unroll")) {
                fprintf(stderr, "  (count=%u, draw %u)\n", count, d);
                return 1;
            }
            mark_release(m[d]);
        }
    }
    vx_buffer_release(cnt.buf);
    return 0;
}

// x = ceil(((bytes - bias) / div) / 2^shift); a bias above the count skips.
int test_byte_count() {
    Words rec;
    if (words_create(rec, 64)) {
        return 1;
    }
    struct Case { uint32_t bytes, bias, div, shift, gx; };
    const Case cases[] = {
        {1000, 16, 7, 2, 35},   // 984 / 7 = 140, ceil(140 / 4) = 35
        {1000, 16, 7, 3, 18},   // ceil(140 / 8) = 18
        {13,   3,  1, 1, 5},    // divisor 1 bypasses: ceil(10 / 2) = 5
        {100,  0,  0, 0, 100},
        {20,   0,  7, 0, 2},    // 20 / 7 = 2, floor
        {6,    0,  7, 0, 0},    // 6 / 7 = 0: skipped
        {9,    10, 3, 0, 0},    // bias above the count: skipped
    };
    for (const Case& c : cases) {
        if (words_write(rec, 0, {c.bytes})) {
            return 1;
        }
        Mark m;
        if (mark_create(m) || mark_clear(m)) {
            return 1;
        }
        vx_launch_info_t l = mark_launch(m, 1, 0, 0, 0);
        vx_launch_indirect_t x = indirect();
        x.record_addr   = rec.addr;
        x.x_bias        = c.bias;
        x.x_div         = c.div;
        x.grid_shift[0] = uint8_t(c.shift);
        l.next = &x;
        CHECK_VX(submit({launch_cmd(&l)}));
        if (mark_expect(m, c.gx ? 1 : 0, c.gx, 1, 1, "byte count")) {
            fprintf(stderr, "  (bytes=%u bias=%u div=%u shift=%u)\n", c.bytes, c.bias, c.div, c.shift);
            return 1;
        }
        mark_release(m);
    }
    vx_buffer_release(rec.buf);
    return 0;
}

// Record words on both sides of a 64-byte line, and the highest word index.
int test_line_crossing() {
    Words rec;
    if (words_create(rec, 256)) {
        return 1;
    }
    std::vector<uint32_t> w(64, 0);
    // record at +56: words 0,1 on line 0; word 2 on line 1; word 14 at +112.
    w[56 / 4 + 0]  = 4;
    w[56 / 4 + 1]  = 3;
    w[56 / 4 + 2]  = 2;
    w[56 / 4 + 14] = 7;
    if (words_write(rec, 0, w)) {
        return 1;
    }
    Mark m, m14;
    if (mark_create(m) || mark_create(m14) || mark_clear(m) || mark_clear(m14)) {
        return 1;
    }
    vx_launch_info_t l = mark_launch(m, 3, 0, 0, 0);
    vx_launch_indirect_t x = indirect();
    x.record_addr = rec.addr + 56;
    l.next = &x;
    vx_launch_info_t l14 = mark_launch(m14, 2, 0, 0, 0);
    vx_launch_indirect_t x14 = indirect();
    x14.record_addr  = rec.addr + 56;
    x14.grid_word[0] = 14;
    x14.grid_word[1] = 2;
    l14.next = &x14;
    CHECK_VX(submit({launch_cmd(&l), launch_cmd(&l14)}));
    if (mark_expect(m, 1, 4, 3, 2, "line-crossing record") ||
        mark_expect(m14, 1, 7, 2, 1, "record word 14")) {
        return 1;
    }
    mark_release(m);
    mark_release(m14);
    vx_buffer_release(rec.buf);
    return 0;
}

int test_enqueue_launch() {
    Words rec;
    if (words_create(rec, 64) || words_write(rec, 0, {6, 2, 1})) {
        return 1;
    }
    Mark m;
    if (mark_create(m) || mark_clear(m)) {
        return 1;
    }
    vx_launch_info_t l = mark_launch(m, 2, 0, 0, 0);
    vx_launch_indirect_t x = indirect();
    x.record_addr = rec.addr;
    l.next = &x;
    CHECK_VX(vx_enqueue_launch(g_q, &l, 0, nullptr, nullptr));
    CHECK_VX(vx_queue_finish(g_q, VX_TIMEOUT_INFINITE));
    if (mark_expect(m, 1, 6, 2, 1, "vx_enqueue_launch")) {
        return 1;
    }
    mark_release(m);
    vx_buffer_release(rec.buf);
    return 0;
}

int test_validation() {
    Mark m;
    if (mark_create(m)) {
        return 1;
    }
    vx_launch_info_t l = mark_launch(m, 1, 1, 1, 1);
    vx_launch_info_t l2 = mark_launch(m, 1, 1, 1, 1);
    struct Bad {
        const char* what;
        void (*mutate)(vx_launch_indirect_t&, vx_launch_info_t&);
        vx_result_t want;
    };
    const Bad bad[] = {
        {"skip past the end",
         [](vx_launch_indirect_t& x, vx_launch_info_t&) { x.skip_count = 2; },
         VX_ERR_INVALID_VALUE},
        {"grid word 16",
         [](vx_launch_indirect_t& x, vx_launch_info_t&) { x.record_addr = 64; x.grid_word[0] = 16; },
         VX_ERR_INVALID_VALUE},
        {"grid shift 16",
         [](vx_launch_indirect_t& x, vx_launch_info_t&) { x.record_addr = 64; x.grid_shift[0] = 16; },
         VX_ERR_INVALID_VALUE},
        {"misaligned record",
         [](vx_launch_indirect_t& x, vx_launch_info_t&) { x.record_addr = 66; },
         VX_ERR_INVALID_VALUE},
        {"misaligned predicate",
         [](vx_launch_indirect_t& x, vx_launch_info_t&) { x.pred_addr = 65; },
         VX_ERR_INVALID_VALUE},
        {"unknown flag",
         [](vx_launch_indirect_t& x, vx_launch_info_t&) { x.flags = 1u << 7; },
         VX_ERR_INVALID_VALUE},
        {"predicate-only with a kernel",
         [](vx_launch_indirect_t& x, vx_launch_info_t&) { x.flags = VX_INDIRECT_NO_LAUNCH; },
         VX_ERR_INVALID_VALUE},
        {"record grid without ndim",
         [](vx_launch_indirect_t& x, vx_launch_info_t& li) { x.record_addr = 64; li.ndim = 0; },
         VX_ERR_INVALID_VALUE},
        {"record grid with a cluster",
         [](vx_launch_indirect_t& x, vx_launch_info_t& li) {
             x.record_addr = 64; li.grid_dim[0] = 2; li.cluster_dim[0] = 2; },
         VX_ERR_INVALID_VALUE},
        {"wrong struct type",
         [](vx_launch_indirect_t& x, vx_launch_info_t&) { x.type = vx_struct_type_e(99); },
         VX_ERR_INVALID_INFO},
        {"short struct",
         [](vx_launch_indirect_t& x, vx_launch_info_t&) { x.struct_size = 8; },
         VX_ERR_INVALID_INFO},
    };
    for (const Bad& b : bad) {
        vx_launch_info_t li = l;
        vx_launch_indirect_t x = indirect();
        b.mutate(x, li);
        li.next = &x;
        vx_result_t r = vx_enqueue_commands(g_q, std::vector<vx_command_t>{
            launch_cmd(&li), launch_cmd(&l2)}.data(), 2, 0, nullptr, nullptr);
        EXPECT(r == b.want, "%s: returned %s, expected %s", b.what,
               vx_result_string(r), vx_result_string(b.want));
    }
    // A block larger than a core, or more shared memory than a core has, is
    // refused on every launch path, extended or not.
    for (int k = 0; k < 2; ++k) {
        vx_launch_info_t li = l;
        if (k == 0) {
            li.block_dim[0] = 1u << 20;
        } else {
            li.lmem_size = 1u << 30;
        }
        vx_launch_indirect_t x = indirect();
        li.next = &x;
        const char* what = k ? "oversized lmem" : "oversized block";
        vx_result_t r = vx_enqueue_launch(g_q, &li, 0, nullptr, nullptr);
        EXPECT(r == VX_ERR_INVALID_VALUE, "%s on extended vx_enqueue_launch: returned %s",
               what, vx_result_string(r));
        li.next = nullptr;
        vx_command_t c = launch_cmd(&li);
        r = vx_enqueue_commands(g_q, &c, 1, 0, nullptr, nullptr);
        EXPECT(r == VX_ERR_INVALID_VALUE, "%s on vx_enqueue_commands: returned %s",
               what, vx_result_string(r));
        r = vx_enqueue_draw(g_q, &c, 1, 0, nullptr, nullptr);
        EXPECT(r == VX_ERR_INVALID_VALUE, "%s on vx_enqueue_draw: returned %s",
               what, vx_result_string(r));
    }
    // A skip range on vx_enqueue_launch has nothing to skip.
    vx_launch_info_t li = l;
    vx_launch_indirect_t x = indirect();
    x.skip_count = 1;
    li.next = &x;
    vx_result_t r = vx_enqueue_launch(g_q, &li, 0, nullptr, nullptr);
    EXPECT(r == VX_ERR_INVALID_VALUE, "skip on vx_enqueue_launch: returned %s",
           vx_result_string(r));
    CHECK_VX(vx_queue_finish(g_q, VX_TIMEOUT_INFINITE));
    mark_release(m);
    return 0;
}

} // namespace

int main(int argc, char** argv) {
    std::string vxbin;
    for (int i = 1; i < argc; ++i) {
        if (std::string(argv[i]) == "-k" && i + 1 < argc) {
            vxbin = argv[++i];
        }
    }
    if (vxbin.empty()) {
        if (const char* env = std::getenv("VX_TEST_INDIRECT_VXBIN")) {
            vxbin = env;
        }
    }
    if (vxbin.empty()) {
        fprintf(stderr, "FAILED: no kernel image (set VX_TEST_INDIRECT_VXBIN or -k)\n");
        return 1;
    }

    CHECK_VX(vx_device_open(0, &g_dev));
    CHECK_VX(vx_queue_create(g_dev, nullptr, &g_q));
    uint64_t nt = 0;
    CHECK_VX(vx_device_query(g_dev, VX_CAPS_NUM_THREADS, &nt));
    g_nt = uint32_t(nt);
    vx_module_h mod = nullptr;
    CHECK_VX(vx_module_load_file(g_dev, vxbin.c_str(), &mod));
    CHECK_VX(vx_module_get_kernel(mod, "produce_k", &g_produce));
    CHECK_VX(vx_module_get_kernel(mod, "mark_k", &g_mark));

    struct Test { const char* name; int (*fn)(); };
    const Test tests[] = {
        {"caps",                 test_caps},
        {"device-written grid",  test_device_grid},
        {"zero dimensions",      test_zero_dims},
        {"predicate",            test_predicate},
        {"skip range (ring)",    [] { return test_skip_range(false); }},
        {"skip range (bundle)",  [] { return test_skip_range(true); }},
        {"skip into a guard (ring)",   [] { return test_skip_into_guard(false); }},
        {"skip into a guard (bundle)", [] { return test_skip_into_guard(true); }},
        {"predicate-only (ring)",   [] { return test_no_launch(false); }},
        {"predicate-only (bundle)", [] { return test_no_launch(true); }},
        {"long draw bundle",     test_long_bundle},
        {"count unroll",         test_count_unroll},
        {"byte count",           test_byte_count},
        {"line-crossing record", test_line_crossing},
        {"vx_enqueue_launch",    test_enqueue_launch},
        {"validation",           test_validation},
    };
    int failures = 0;
    for (const Test& t : tests) {
        printf("[ RUN  ] %s\n", t.name);
        fflush(stdout);
        if (t.fn() != 0) {
            printf("[ FAIL ] %s\n", t.name);
            ++failures;
        } else {
            printf("[  OK  ] %s\n", t.name);
        }
        fflush(stdout);
    }

    vx_kernel_release(g_mark);
    vx_kernel_release(g_produce);
    vx_module_release(mod);
    vx_queue_release(g_q);
    vx_device_release(g_dev);
    if (failures) {
        printf("FAILED! (%d sections)\n", failures);
        return 1;
    }
    printf("PASSED!\n");
    return 0;
}
