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

import sys
import argparse
import csv
import re
import inspect

configs = None

class PerfCounter:
    def __init__(self, name):
        self.name  = name
        self.total = 0
        self.count = 0
        self.min   = 0
        self.max   = 0
        self.min_uuid = None
        self.max_uuid = None

    def update(self, uuid, value):
        if self.count != 0:
            if value < self.min:
                self.min = value
                self.min_uuid = uuid
            if value > self.max:
                self.max = value
                self.max_uuid = uuid
        else:
            self.min = value
            self.max = value
            self.min_uuid = uuid
            self.max_uuid = uuid
        self.total = self.total + value
        self.count = self.count + 1

    def dump(self):
        if self.count != 0:
            avg = self.total // self.count
        else:
            avg = 0
        print("{} latency: avg={}, min={} (#{}), max={} (#{})".format(self.name, avg, self.min, self.min_uuid, self.max, self.max_uuid))

def load_config(filename):
    config_pattern = r"CONFIGS: num_threads=(\d+), num_warps=(\d+), num_cores=(\d+), num_clusters=(\d+), socket_size=(\d+), local_mem_base=0x([0-9a-fA-F]+), num_barriers=(\d+)"
    with open(filename, 'r') as file:
        for line in file:
            config_match = re.search(config_pattern, line)
            if config_match:
                config = {
                    'num_threads': int(config_match.group(1)),
                    'num_warps': int(config_match.group(2)),
                    'num_cores': int(config_match.group(3)),
                    'num_clusters': int(config_match.group(4)),
                    'socket_size': int(config_match.group(5)),
                    'local_mem_base': int(config_match.group(6), 16),
                    'num_barriers': int(config_match.group(7)),
                }
                return config
    print("Error: missing CONFIGS: header")
    sys.exit(1)

def parse_simx(log_lines):
    # Regex for standard TRACE lines
    line_pattern = r"^TRACE\s+(\d+):\s+([a-zA-Z0-9_-]+)\s+([a-zA-Z0-9_-]+):\s+(.*)$"

    # Regex for DEBUG lines (contain the data values)
    # Some instructions have no operands and are logged as "DEBUG Instr: FENCE, ...",
    # so allow an optional trailing comma after the opcode.
    debug_instr_pattern = r"DEBUG Instr:\s+([a-zA-Z0-9_\.]+),?\s+.*#(\d+)"
    debug_src_pattern = r"DEBUG Src\d+ Reg:\s+(.+)"
    debug_dest_pattern = r"DEBUG Dest Reg:\s+(.+)"

    # Arg patterns for TRACE lines
    uuid_pattern = r"#(\d+)"
    cid_pattern = r"cid=(\d+)"
    wid_pattern = r"wid=(\d+)"
    tmask_pattern = r"tmask=([01]+)"
    pc_pattern = r"PC=(0x[0-9a-fA-F]+)"
    op_pattern = r"op=([a-zA-Z0-9_\.]+)"
    rd_pattern = r"rd=([a-zA-Z0-9]+)"
    rs_pattern = r"rs\d+=([a-zA-Z0-9]+)"

    entries = []
    instr_data = {}

    perf_sched = PerfCounter("Schedule")
    perf_issue = PerfCounter("Issue")
    perf_exec  = PerfCounter("Execute")

    schd_ticks = {}
    op_ticks = {}

    current_debug_uuid = None

    for lineno, line in enumerate(log_lines, start=1):
        try:
            # --- DEBUG Line Parsing (Captures Data Values) ---
            if line.startswith("DEBUG"):
                # 1. Identify Instruction & Opcode
                instr_match = re.search(debug_instr_pattern, line)
                if instr_match:
                    opcode = instr_match.group(1)
                    uuid = int(instr_match.group(2))
                    current_debug_uuid = uuid

                    if uuid not in instr_data:
                        instr_data[uuid] = {}

                    instr_data[uuid]["opcode"] = opcode
                    instr_data[uuid]["lineno"] = lineno
                    continue

                # 2. Capture Source/Dest Registers (with values)
                if current_debug_uuid is not None:
                    # Capture Src (Operands)
                    src_match = re.search(debug_src_pattern, line)
                    if src_match:
                        operand_str = src_match.group(1) # e.g. x0={0x0, ...}
                        if "operands_list" not in instr_data[current_debug_uuid]:
                            instr_data[current_debug_uuid]["operands_list"] = []
                        instr_data[current_debug_uuid]["operands_list"].append(operand_str)
                        continue

                    # Capture Dest (Destination)
                    dest_match = re.search(debug_dest_pattern, line)
                    if dest_match:
                        dest_str = dest_match.group(1) # e.g. x5={0x4, ...}
                        instr_data[current_debug_uuid]["destination"] = dest_str
                        continue

            # --- TRACE Line Parsing (Captures Pipeline Timing) ---
            line_match = re.search(line_pattern, line)
            if not line_match:
                continue

            timestamp = int(line_match.group(1))
            module = line_match.group(2)
            action = line_match.group(3)
            args_str = line_match.group(4)

            uuid_match = re.search(uuid_pattern, args_str)
            if not uuid_match:
                continue
            uuid = int(uuid_match.group(1))

            # --- Stage Identification ---

            # 1. Schedule: Create/Update entry
            if action == "schedule":
                if uuid not in instr_data:
                    instr_data[uuid] = {}
                trace = instr_data[uuid]

                # Update basic info
                trace["lineno"] = lineno
                trace["uuid"] = uuid

                pc_m = re.search(pc_pattern, args_str)
                cid_m = re.search(cid_pattern, args_str)
                wid_m = re.search(wid_pattern, args_str)
                tmask_m = re.search(tmask_pattern, args_str)

                trace["PC"] = pc_m.group(1) if pc_m else "0x0"
                trace["core_id"] = int(cid_m.group(1)) if cid_m else 0
                trace["warp_id"] = int(wid_m.group(1)) if wid_m else 0
                trace["tmask"] = tmask_m.group(1) if tmask_m else "0000"

                schd_ticks[uuid] = timestamp

            # 2. Operands: Issue timing
            elif action == "operands":
                op_ticks[uuid] = timestamp
                if uuid in schd_ticks:
                    cycles = timestamp - schd_ticks[uuid]
                    perf_sched.update(uuid, cycles)

            # 3. Execute: Update Opcode (optional)
            elif action == "execute":
                # TRACE opcode is sometimes more accurate for execution units, but DEBUG is usually fine.
                pass

            # 4. Commit: Finalize
            elif action == "commit":
                if uuid in instr_data:
                    trace = instr_data[uuid]

                    start_exec_tick = op_ticks.get(uuid, schd_ticks.get(uuid, timestamp))
                    cycles = timestamp - start_exec_tick
                    perf_exec.update(uuid, cycles)

                    # Finalize operands list into string
                    if "operands_list" in trace:
                        trace["operands"] = ", ".join(trace["operands_list"])
                        del trace["operands_list"]

                    # Ensure required fields exist
                    if "destination" not in trace: trace["destination"] = ""
                    if "operands" not in trace: trace["operands"] = ""

                    entries.append(trace)

                    # Cleanup
                    del instr_data[uuid]
                    if uuid in schd_ticks: del schd_ticks[uuid]
                    if uuid in op_ticks: del op_ticks[uuid]

        except Exception as e:
            # print("Error parsing line {}: {} | {}".format(lineno, e, line.strip()))
            pass

    perf_sched.dump()
    perf_issue.dump()
    perf_exec.dump()
    return entries

def reverse_binary(bin_str):
    return bin_str[::-1]

def bin_to_array(bin_str):
    return [int(bit) for bit in bin_str]

def append_reg(text, reg, sep):
    if sep:
        text += ", "
    ireg = int(reg)
    rtype = ireg // 32
    rvalue = ireg % 32
    if (rtype == 2):
        text += "v" + str(rvalue)
    elif (rtype == 1):
        text += "f" + str(rvalue)
    else:
        text += "x" + str(rvalue)
    sep = True
    return text, sep

def reg_value(rtype, value):
    if rtype == 1:
        ivalue = int(value, 16)
        ivalue32 = ivalue & 0xFFFFFFFF
        return "0x{:x}".format(ivalue32)
    else:
        return value

def append_value(text, reg, value, tmask_arr, sep):
    text, sep = append_reg(text, reg, sep)
    ireg = int(reg)
    rtype = ireg // 32
    text += "={"
    for i in range(len(tmask_arr)):
        if i != 0:
            text += ", "
        if tmask_arr[i]:
            text += reg_value(rtype, value[i])
        else:
            text +="-"
    text += "}"
    return text, sep

def simd_data(sub_array, index, count, default=0):
    size = len(sub_array)
    total_subsets = count // size
    new_array = [default] * count
    start_index = index * size
    if start_index + size <= count:
        new_array[start_index:start_index + size] = sub_array
    return new_array

def merge_data(trace, key, new_data, mask):
    if key in trace:
        merged_data = trace[key]
        for i in range(len(mask)):
            if mask[i] == 1:
                merged_data[i] = new_data[i]
        trace[key] = merged_data
    else:
        trace[key] = new_data

def parse_rtlsim(log_lines):
    global configs
    # Regex to capture timestamp, topology, module, and action
    line_pattern = r"(\d+):\s+cluster(\d+)-socket(\d+)-core(\d+)-([a-zA-Z0-9_-]+)\s+([a-zA-Z0-9_-]+):"

    pc_pattern = r"PC=(0x[0-9a-fA-F]+)"
    op_pattern = r"op=([\?0-9a-zA-Z_\.]+)"
    warp_id_pattern = r"wid=(\d+)"
    tmask_pattern = r"tmask=(\d+)"
    wb_pattern = r"wb=(\d)"

    # Updated patterns for new format
    used_rs_pattern = r"used_rs=([01]+)"
    sid_pattern = r"sid=(\d+)"
    rd_pattern = r"rd=([xz\d]+)"
    rs1_pattern = r"rs1=([xz\d]+)"
    rs2_pattern = r"rs2=([xz\d]+)"
    rs3_pattern = r"rs3=([xz\d]+)"
    rs1_data_pattern = r"rs1_data=\{(.+?)\}"
    rs2_data_pattern = r"rs2_data=\{(.+?)\}"
    rs3_data_pattern = r"rs3_data=\{(.+?)\}"
    rd_data_pattern = r"data=\{(.+?)\}"
    eop_pattern = r"eop=(\d)"
    uuid_pattern = r"\(#(\d+)\)"

    entries = []
    instr_data = {}
    num_cores = configs['num_cores']
    socket_size = configs['socket_size']
    num_threads = configs['num_threads']
    num_sockets = (num_cores + socket_size - 1) // socket_size
    schd_ticks = {}
    perf_sched = PerfCounter("Schedule")
    perf_issue = PerfCounter("Issue")
    perf_exec  = PerfCounter("Execute")

    for lineno, line in enumerate(log_lines, start=1):
        try:
            line_match = re.search(line_pattern, line)
            if line_match:
                timestamp = int(line_match.group(1))
                cluster_id = int(line_match.group(2))
                socket_id = int(line_match.group(3))
                core_id = int(line_match.group(4))
                module = line_match.group(5)
                action = line_match.group(6)

                uuid_match = re.search(uuid_pattern, line)
                if not uuid_match:
                    continue
                uuid = int(uuid_match.group(1))

                # Pipeline Stage Identification
                is_schedule = "scheduler" in module and action == "dispatch"
                is_decode   = ("decode" in module or "ibuffer-uop" in module) and action == "decode"
                is_dispatch = "dispatcher" in module and action == "dispatch"
                is_commit   = "commit" in module and action == "commit"

                if is_schedule:
                    schd_ticks[uuid] = timestamp

                elif is_decode:
                    trace = {}
                    trace["uuid"] = uuid
                    trace["PC"] = re.search(pc_pattern, line).group(1)
                    trace["core_id"] = ((((cluster_id * num_sockets) + socket_id) * socket_size) + core_id)
                    trace["warp_id"] = int(re.search(warp_id_pattern, line).group(1))
                    trace["tmask"] = reverse_binary(re.search(tmask_pattern, line).group(1))
                    trace["opcode"] = re.search(op_pattern, line).group(1)

                    # Parse used_rs (binary string)
                    used_rs_str = re.search(used_rs_pattern, line).group(1)
                    trace["used_rs"] = bin_to_array(reverse_binary(used_rs_str))

                    # Safely extract registers (some instrs might omit unused registers)
                    rd_match = re.search(rd_pattern, line)
                    trace["rd"] = rd_match.group(1) if rd_match else "0"

                    rs1_match = re.search(rs1_pattern, line)
                    trace["rs1"] = rs1_match.group(1) if rs1_match else "0"

                    rs2_match = re.search(rs2_pattern, line)
                    trace["rs2"] = rs2_match.group(1) if rs2_match else "0"

                    rs3_match = re.search(rs3_pattern, line)
                    trace["rs3"] = rs3_match.group(1) if rs3_match else "0"

                    trace["ibuf_ticks"] = timestamp
                    instr_data[uuid] = trace

                    if uuid in schd_ticks:
                        ticks = schd_ticks[uuid]
                        cycles = (timestamp - ticks + 1) // 2
                        perf_sched.update(uuid, cycles)

                elif is_dispatch:
                    if uuid in instr_data:
                        trace = instr_data[uuid]
                        sid = int(re.search(sid_pattern, line).group(1))
                        curr_tmask = re.search(tmask_pattern, line).group(1)
                        src_tmask_arr = simd_data(bin_to_array(curr_tmask)[::-1], sid, num_threads, 0)

                        trace["lineno"] = lineno
                        used_rs = trace["used_rs"]
                        if used_rs[0]:
                            merge_data(trace, 'rs1_data', simd_data(re.search(rs1_data_pattern, line).group(1).split(', ')[::-1], sid, num_threads, '0x0'), src_tmask_arr)
                        if used_rs[1]:
                            merge_data(trace, 'rs2_data', simd_data(re.search(rs2_data_pattern, line).group(1).split(', ')[::-1], sid, num_threads, '0x0'), src_tmask_arr)
                        if used_rs[2]:
                            merge_data(trace, 'rs3_data', simd_data(re.search(rs3_data_pattern, line).group(1).split(', ')[::-1], sid, num_threads, '0x0'), src_tmask_arr)

                        trace["issued"] = True
                        trace["issue_ticks"] = timestamp
                        instr_data[uuid] = trace
                        cycles = (timestamp - trace["ibuf_ticks"] + 1) // 2
                        perf_issue.update(uuid, cycles)

                elif is_commit:
                    if uuid in instr_data:
                        trace = instr_data[uuid]
                        if "issued" in trace:
                            sid = int(re.search(sid_pattern, line).group(1))
                            used_rs = trace["used_rs"]
                            curr_tmask = re.search(tmask_pattern, line).group(1)
                            dst_tmask_arr = simd_data(bin_to_array(curr_tmask)[::-1], sid, num_threads, 0)

                            wb = re.search(wb_pattern, line).group(1) == "1"
                            if wb:
                                merge_data(trace, 'rd_data', simd_data(re.search(rd_data_pattern, line).group(1).split(', ')[::-1], sid, num_threads, '0x0'), dst_tmask_arr)

                            instr_data[uuid] = trace
                            eop = re.search(eop_pattern, line).group(1) == "1"
                            if eop:
                                tmask_arr = bin_to_array(trace["tmask"])
                                destination = ''
                                if wb and 'rd_data' in trace:
                                    destination, sep = append_value(destination, trace["rd"], trace['rd_data'], tmask_arr, False)
                                    del trace['rd_data']
                                trace["destination"] = destination

                                operands = ''
                                sep = False
                                if used_rs[0] and "rs1_data" in trace:
                                    operands, sep = append_value(operands, trace["rs1"], trace["rs1_data"], tmask_arr, sep)
                                    del trace["rs1_data"]
                                if used_rs[1] and "rs2_data" in trace:
                                    operands, sep = append_value(operands, trace["rs2"], trace["rs2_data"], tmask_arr, sep)
                                    del trace["rs2_data"]
                                if used_rs[2] and "rs3_data" in trace:
                                    operands, sep = append_value(operands, trace["rs3"], trace["rs3_data"], tmask_arr, sep)
                                    del trace["rs3_data"]
                                trace["operands"] = operands

                                cycles = (timestamp - trace["issue_ticks"] + 1) // 2
                                perf_exec.update(uuid, cycles)

                                # Clean up
                                for k in ["ibuf_ticks", "issue_ticks", "used_rs", "rd", "rs1", "rs2", "rs3", "issued"]:
                                    if k in trace: del trace[k]

                                del instr_data[uuid]
                                entries.append(trace)
        except Exception as e:
            print("Error: {0} ({1}); {2}".format(e, lineno, line))

    perf_sched.dump()
    perf_issue.dump()
    perf_exec.dump()
    return entries

def write_csv(sublogs, csv_filename, log_type):
    with open(csv_filename, 'w', newline='') as csv_file:
        fieldnames = ["uuid", "PC", "opcode", "core_id", "warp_id", "tmask", "destination", "operands"]
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()

        for sublog in sublogs:
            entries = None

            # parse sublog
            if log_type == "rtlsim":
                entries = parse_rtlsim(sublog)
            elif log_type == "simx":
                entries = parse_simx(sublog)
            else:
                print('Error: invalid log type')
                sys.exit()

            # sort entries by uuid
            entries.sort(key=lambda x: (int(x['uuid'])))
            for entry in entries:
                del entry['lineno']

            for entry in entries:
                writer.writerow(entry)

def split_log_file(log_filename):
    with open(log_filename, 'r') as log_file:
        log_lines = log_file.readlines()

    sublogs = []
    current_sublog = None

    for line in log_lines:
        if line.startswith("[VXDRV] START"):
            if current_sublog is not None:
                sublogs.append(current_sublog)
            current_sublog = [line]
        elif current_sublog is not None:
            current_sublog.append(line)

    if current_sublog is not None:
        sublogs.append(current_sublog)
    else:
        sublogs.append(log_lines)

    return sublogs

def parse_args():
    parser = argparse.ArgumentParser(description='CPU trace log to CSV format converter.')
    parser.add_argument('-t', '--type', default='simx', help='log type (rtlsim or simx)')
    parser.add_argument('-o', '--csv', default='trace.csv', help='Output CSV file')
    parser.add_argument('log', help='Input log file')
    return parser.parse_args()

def main():
    global configs
    args = parse_args()
    configs = load_config(args.log)
    sublogs = split_log_file(args.log)
    write_csv(sublogs, args.csv, args.type)

if __name__ == "__main__":
    main()
