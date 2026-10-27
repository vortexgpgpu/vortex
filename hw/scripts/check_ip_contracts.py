#!/usr/bin/env python3
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

"""Check the pipeline contracts of the Xilinx floating-point cores.

Each black box in xilinx_ip_stubs.sv declares the latency it is used at
(`fixed_latency`). That latency must be the core's C_Latency in
xilinx_ip_gen.tcl, the latency the RTL around it asserts, its simulation
model's (sim/foldsim/xilinx_fp_models.sv) and, with --cflags, the
VX_CFG_*_LATENCY the configuration resolves to for it.

usage: check_ip_contracts.py [--cflags '<-D flags of the build>']
"""
import argparse
import os
import re
import subprocess
import sys

HERE = os.path.dirname(os.path.realpath(__file__))
ROOT = os.path.realpath(os.path.join(HERE, '..', '..'))

# Where the RTL states the latency it uses each core at.
RTL_ASSERTS = {
    'xil_fma': ('hw/rtl/fpu/VX_fma_unit.sv', r'USE_VENDOR_IP\s*\|\|\s*\(LATENCY\s*==\s*(\d+)\)'),
    'xil_fdiv': ('hw/rtl/fpu/VX_fdiv_unit.sv', r'USE_VENDOR_IP\s*\|\|\s*\(LATENCY\s*==\s*(\d+)\)'),
    'xil_fsqrt': ('hw/rtl/fpu/VX_fsqrt_unit.sv', r'USE_VENDOR_IP\s*\|\|\s*\(LATENCY\s*==\s*(\d+)\)'),
    'xil_fmul': ('hw/rtl/tcu/dsp/VX_tcu_fedp_dsp.sv', r'localparam\s+FMUL_LATENCY\s*=\s*(\d+)'),
    'xil_fadd': ('hw/rtl/tcu/dsp/VX_tcu_fedp_dsp.sv', r'localparam\s+FADD_LATENCY\s*=\s*(\d+)'),
}

# The configuration's latency of each core, when the configuration selects it.
CONFIG_LATENCIES = {
    'xil_fma': [('VX_CFG_FMA_LATENCY', 'VX_CFG_FPU_TYPE_DSP')],
    'xil_fdiv': [('VX_CFG_FDIV_LATENCY', 'VX_CFG_FPU_TYPE_DSP'), ('VX_CFG_RTU_FDIV_LATENCY', None)],
    'xil_fsqrt': [('VX_CFG_FSQRT_LATENCY', 'VX_CFG_FPU_TYPE_DSP')],
}


def stub_contracts(path):
    """module -> its attributes, from the `(* ... *)` before each module."""
    text = open(path).read()
    contracts = {}
    for attrs, name in re.findall(r'\(\*(.*?)\*\)\s*module\s+(\w+)', text, re.S):
        values = {}
        for item in re.findall(r'(\w+)(?:\s*=\s*("[^"]*"|\w+))?', attrs):
            values[item[0]] = item[1].strip('"') if item[1] else ''
        contracts[name] = values
    return contracts


def tcl_latencies(path):
    """module -> C_Latency, from each core's set_property dictionary."""
    text = open(path).read()
    latencies = {}
    for body, name in re.findall(r'set_property -dict \[list (.*?)\] \[get_ips (\w+)\]', text, re.S):
        m = re.search(r'CONFIG\.C_Latency \{(\d+)\}', body)
        latencies[name] = int(m.group(1)) if m else None
    return latencies


def model_latencies(path):
    """module -> the latency of its simulation model's pipeline."""
    text = open(path).read()
    return {name: int(lat) for name, lat in
            re.findall(r'module (\w+) \(.*?xilinx_fp_model_pipe #\(\.LATENCY\((\d+)\)', text, re.S)}


def config_values(cflags):
    out = subprocess.run([sys.executable, os.path.join(ROOT, 'ci', 'gen_config.py'),
                          '--config', os.path.join(ROOT, 'VX_config.toml'), '--cflags', cflags],
                         check=True, capture_output=True, text=True).stdout
    values = {}
    for flag in out.split():
        if flag.startswith('-D'):
            name, _, value = flag[2:].partition('=')
            values[name] = value
    return values


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--cflags', help="the build's -D flags, to check its resolved latencies")
    a = ap.parse_args()
    errors = []
    contracts = stub_contracts(os.path.join(HERE, 'xilinx_ip_stubs.sv'))
    tcl = tcl_latencies(os.path.join(HERE, 'xilinx_ip_gen.tcl'))
    config = config_values(a.cflags) if a.cflags is not None else None
    models = model_latencies(os.path.join(ROOT, 'sim', 'foldsim', 'xilinx_fp_models.sv'))
    for name, rtl in RTL_ASSERTS.items():
        c = contracts.get(name)
        if c is None or 'fixed_latency' not in c:
            errors.append(f'{name}: no fixed_latency contract in xilinx_ip_stubs.sv')
            continue
        latency = int(c['fixed_latency'])
        if tcl.get(name) != latency:
            errors.append(f'{name}: contract latency {latency}, C_Latency {tcl.get(name)} in xilinx_ip_gen.tcl')
        if models.get(name) != latency:
            errors.append(f'{name}: contract latency {latency}, sim/foldsim/xilinx_fp_models.sv models {models.get(name)}')
        path, pattern = rtl
        m = re.search(pattern, open(os.path.join(ROOT, path)).read())
        if not m:
            errors.append(f'{name}: {path} no longer states the latency it uses the core at')
        elif int(m.group(1)) != latency:
            errors.append(f'{name}: contract latency {latency}, {path} uses {m.group(1)}')
        if config is None:
            continue
        for knob, selector in CONFIG_LATENCIES.get(name, []):
            if selector and selector not in config:
                continue
            # The RTU uses the vendor divider on the Vivado flow only.
            if knob == 'VX_CFG_RTU_FDIV_LATENCY' and not re.search(r'-DVIVADO\b', a.cflags):
                continue
            if knob in config and int(config[knob]) != latency:
                errors.append(f'{name}: contract latency {latency}, the configuration resolves {knob}={config[knob]}')
    for e in errors:
        print(f'ERROR: {e}', file=sys.stderr)
    if errors:
        sys.exit(1)
    print(f'xilinx IP contracts: {", ".join(f"{n} {contracts[n]["fixed_latency"]}" for n in RTL_ASSERTS)} consistent')


if __name__ == '__main__':
    main()
