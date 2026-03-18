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

#include <iostream>
#include <unordered_map>
#include <vector>
#include <verilated_vcd_c.h>
#include "VVX_mem_scheduler.h"
#include "ram.h"

#define SIM_TIME 5000

int generate_rand (int min, int max);
int generate_rand_mask (int mask);

class MemSim {
public:
    MemSim();
    virtual ~MemSim();

    void run(RAM *ram);

private:
    VVX_mem_scheduler *msu_;
#ifdef VCD_OUTPUT
    VerilatedVcdC* tfp_;
#endif

    void eval();
    void step();
    void reset();

    void attach_core();
    void attach_ram(RAM *ram);
};
