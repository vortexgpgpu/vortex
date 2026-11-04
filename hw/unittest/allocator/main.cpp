// Copyright © 2019-2026
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

// Functional test for VX_allocator: random acquires and releases (together
// in the same cycle too) against a model that hands out the lowest free slot
// after the cycle's release and acquire; checks acquire_addr, empty and full
// every cycle (acquire_addr only while not full: it is don't-care when full).

#include "VVX_allocator.h"
#include "vl_simulator.h"

#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

#ifndef SIZE
#define SIZE 24
#endif

int main(int argc, char** argv) {
    Verilated::commandArgs(argc, argv);
    size_t cycles = 2000000;
    if (argc > 1) {
        long v = std::strtol(argv[1], nullptr, 10);
        if (v > 0) cycles = static_cast<size_t>(v);
    }

    vl_simulator<VVX_allocator> sim;
    std::mt19937 rng(0xA110Cu);

    std::vector<bool> free_slots(SIZE, true);
    std::vector<int> held;
    int next = 0;
    bool empty = true, full = false;
    auto lowest = [&]() { for (int i = 0; i < SIZE; ++i) if (free_slots[i]) return i; return -1; };

    sim->acquire_en = 0; sim->release_en = 0; sim->release_addr = 0;
    uint64_t ticks = sim.reset(0);

    size_t errors = 0, acquires = 0, releases = 0, both = 0;
    for (size_t c = 0; c < cycles; ++c) {
        if ((!full && int(sim->acquire_addr) != next) || bool(sim->empty) != empty || bool(sim->full) != full) {
            if (errors < 16)
                printf("MISMATCH cycle %zu: addr %u/%d empty %u/%d full %u/%d\n", c,
                       sim->acquire_addr, next, sim->empty, empty, sim->full, full);
            ++errors;
        }
        // Phases of mostly-acquire and mostly-release sweep the occupancy.
        const int phase = (c / 4096) % 3;
        const int p_acq = phase == 0 ? 80 : (phase == 1 ? 30 : 55);
        const bool acq = !full && int(rng() % 100) < p_acq;
        const bool rel = !held.empty() && int(rng() % 100) < (100 - p_acq);
        int raddr = 0;
        if (rel) {
            const size_t k = rng() % held.size();
            raddr = held[k];
            held[k] = held.back();
            held.pop_back();
        }
        sim->acquire_en   = acq;
        sim->release_en   = rel;
        sim->release_addr = raddr;
        ticks = sim.step(ticks, 2);

        // model: release, then acquire the current head, then pick the next head
        if (rel) { free_slots[raddr] = true; ++releases; }
        if (acq) { free_slots[next] = false; held.push_back(next); ++acquires; }
        both += (acq && rel);
        if (acq || (rel && full)) {
            const int l = lowest();
            if (l >= 0) next = l;
        }
        empty = true; for (bool f : free_slots) empty &= f;
        full = (lowest() < 0);
    }
    printf("allocator SIZE=%d: cycles=%zu acquires=%zu releases=%zu both=%zu errors=%zu\n%s\n",
           SIZE, cycles, acquires, releases, both, errors, errors ? "FAILED" : "PASSED");
    return errors ? 1 : 0;
}
