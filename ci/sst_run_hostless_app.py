# Copyright © 2019-2023
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Hostless SST runner: instantiate a single vortex.VortexGPGPU
# component and run the given kernel. SST runs Vortex co-resident in
# one process, primes the KMU DCRs directly via proc_->dcr_write
# inside sim/simx/sst/vortex_simulator.cpp, and ticks the simulation
# to completion. No host CPU, no CP, no PIO/DMA.
#
# Hostless is the only mode the SST integration currently supports:
# there is no SST CPU component (e.g. Ariel/Vanadis) wired to a
# Vortex regression test binary today. A future ci/sst_run_app.py
# could add that path; the name slot is reserved.
#
# For memHierarchy timing modeling, the VortexGPGPU component exposes
# an optional `memIface` SubComponent slot — see
# docs/proposals/sst_simx_v3_proposal.md for the wiring recipe.
#
# Configurable via env vars (parallel to ci/gem5_run_hostless_app.py):
#   VORTEX_TEST_DIR    — directory containing the kernel .vxbin
#   VORTEX_TEST_KERNEL — kernel filename inside that dir
#                        (default: kernel.vxbin, matching the
#                         regression-test convention)
#
# Run via:
#   VORTEX_TEST_DIR=tests/kernel/hello VORTEX_TEST_KERNEL=hello.vxbin \
#       sst ci/sst_run_hostless_app.py

import os
import sst

TEST_DIR    = os.environ.get("VORTEX_TEST_DIR")
TEST_KERNEL = os.environ.get("VORTEX_TEST_KERNEL", "kernel.vxbin")
if not TEST_DIR:
    raise RuntimeError("VORTEX_TEST_DIR env var is required")

PROGRAM = f"{TEST_DIR}/{TEST_KERNEL}"

gpu = sst.Component("gpu0", "vortex.VortexGPGPU")
gpu.addParams({
    "clock":   "1GHz",
    "program": PROGRAM,
})
