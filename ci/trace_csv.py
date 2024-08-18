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

def parse_args():
    parser = argparse.ArgumentParser(description='CPU trace log to CSV format converter.')
    parser.add_argument('-t', '--type', default='simx', help='log type (rtlsim or simx)')
    parser.add_argument('-o', '--csv', default='trace.csv', help='Output CSV file')
    parser.add_argument('log', help='Input log file')
    return parser.parse_args()

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
    return None

def parse_simx(log_lines):
    pc_pattern = r"PC=(0x[0-9a-fA-F]+)"
    instr_pattern = r"Instr (0x[0-9a-fA-F]+):"
    opcode_pattern = r"Instr 0x[0-9a-fA-F]+: ([0-9a-zA-Z_\.]+)"
    core_id_pattern = r"cid=(\d+)"
    warp_id_pattern = r"wid=(\d+)"
    tmask_pattern = r"tmask=(\d+)"
    operands_pattern = r"Src\d+ Reg: (.+)"
    destination_pattern = r"Dest Reg: (.+)"
    uuid_pattern = r"#(\d+)"
    entries = []
    instr_data = None
    for lineno, line in enumerate(log_lines, start=1):
        try:
            if line.startswith("DEBUG Fetch:"):
                if instr_data:
                    entries.append(instr_data)
                instr_data = {}
                instr_data["lineno"] = lineno
                instr_data["PC"] = re.search(pc_pattern, line).group(1)
                instr_data["core_id"] = int(re.search(core_id_pattern, line).group(1))
                instr_data["warp_id"] = int(re.search(warp_id_pattern, line).group(1))
                instr_data["tmask"] = re.search(tmask_pattern, line).group(1)
                instr_data["uuid"] = int(re.search(uuid_pattern, line).group(1))
            elif line.startswith("DEBUG Instr"):
                instr_data["instr"] = re.search(instr_pattern, line).group(1)
                instr_data["opcode"] = re.search(opcode_pattern, line).group(1)
            elif line.startswith("DEBUG Src"):
                src_reg = re.search(operands_pattern, line).group(1)
                instr_data["operands"] = (instr_data["operands"] + ', ' + src_reg) if 'operands' in instr_data else src_reg
            elif line.startswith("DEBUG Dest"):
                instr_data["destination"] = re.search(destination_pattern, line).group(1)
        except Exception as e:
            print("Error at line {}: {}".format(lineno, e))
            instr_data = None
    if instr_data:
        entries.append(instr_data)
    return entries

def reverse_binary(bin_str):
    return bin_str[::-1]

def bin_to_array(bin_str):
    return [int(bit) for bit in bin_str]

def append_reg(text, value, sep):
    if sep:
        text += ", "
    ivalue = int(value)
    if (ivalue >= 32):
        text += "f" + str(ivalue % 32)
    else:
        text += "x" + value
    sep = True
    return text, sep

def append_value(text, reg, value, tmask_arr, sep):
    text, sep = append_reg(text, reg, sep)
    text += "={"
    for i in range(len(tmask_arr)):
        if i != 0:
            text += ", "
        if tmask_arr[i]:
            text += value[i]
        else:
            text +="-"
    text += "}"
    return text, sep

def parse_rtlsim(log_lines):
    global configs
    line_pattern = r"\d+: cluster(\d+)-socket(\d+)-core(\d+)-(decode|issue|commit)"
    pc_pattern = r"PC=(0x[0-9a-fA-F]+)"
    instr_pattern = r"instr=(0x[0-9a-fA-F]+)"
    ex_pattern = r"ex=([a-zA-Z]+)"
    op_pattern = r"op=([\?0-9a-zA-Z_\.]+)"
    warp_id_pattern = r"wid=(\d+)"
    tmask_pattern = r"tmask=(\d+)"
    wb_pattern = r"wb=(\d)"
    opds_pattern = r"opds=(\d+)"
    rd_pattern = r"rd=(\d+)"
    rs1_pattern = r"rs1=(\d+)"
    rs2_pattern = r"rs2=(\d+)"
    rs3_pattern = r"rs3=(\d+)"
    rs1_data_pattern = r"rs1_data=\{(.+?)\}"
    rs2_data_pattern = r"rs2_data=\{(.+?)\}"
    rs3_data_pattern = r"rs3_data=\{(.+?)\}"
    rd_data_pattern = r"data=\{(.+?)\}"
    eop_pattern = r"eop=(\d)"
    uuid_pattern = r"#(\d+)"
    entries = []
    instr_data = {}
    num_cores = configs['num_cores']
    socket_size = configs['socket_size']
    num_sockets = (num_cores + socket_size - 1) // socket_size
    for lineno, line in enumerate(log_lines, start=1):
        try:
            line_match = re.search(line_pattern, line)
            if line_match:
                PC = re.search(pc_pattern, line).group(1)
                warp_id = int(re.search(warp_id_pattern, line).group(1))
                tmask = re.search(tmask_pattern, line).group(1)
                uuid = int(re.search(uuid_pattern, line).group(1))
                cluster_id = int(line_match.group(1))
                socket_id = int(line_match.group(2))
                core_id = int(line_match.group(3))
                stage = line_match.group(4)
                if stage == "decode":
                    trace = {}
                    trace["uuid"] = uuid
                    trace["PC"] = PC
                    trace["core_id"] = ((((cluster_id * num_sockets) + socket_id) * socket_size) + core_id)
                    trace["warp_id"] = warp_id
                    trace["tmask"] = reverse_binary(tmask)
                    trace["instr"] = re.search(instr_pattern, line).group(1)
                    trace["opcode"] = re.search(op_pattern, line).group(1)
                    trace["opds"] = bin_to_array(re.search(opds_pattern, line).group(1))
                    trace["rd"] = re.search(rd_pattern, line).group(1)
                    trace["rs1"] = re.search(rs1_pattern, line).group(1)
                    trace["rs2"] = re.search(rs2_pattern, line).group(1)
                    trace["rs3"] = re.search(rs3_pattern, line).group(1)
                    instr_data[uuid] = trace
                elif stage == "issue":
                    if uuid in instr_data:
                        trace = instr_data[uuid]
                        trace["lineno"] = lineno
                        opds = trace["opds"]
                        if opds[1]:
                            trace["rs1_data"] = re.search(rs1_data_pattern, line).group(1).split(', ')[::-1]
                        if opds[2]:
                            trace["rs2_data"] = re.search(rs2_data_pattern, line).group(1).split(', ')[::-1]
                        if opds[3]:
                            trace["rs3_data"] = re.search(rs3_data_pattern, line).group(1).split(', ')[::-1]
                        trace["issued"] = True
                        instr_data[uuid] = trace
                elif stage == "commit":
                    if uuid in instr_data:
                        trace = instr_data[uuid]
                        if "issued" in trace:
                            opds = trace["opds"]
                            dst_tmask_arr = bin_to_array(tmask)[::-1]
                            wb = re.search(wb_pattern, line).group(1) == "1"
                            if wb:
                                rd_data = re.search(rd_data_pattern, line).group(1).split(', ')[::-1]
                                if 'rd_data' in trace:
                                    merged_rd_data = trace['rd_data']
                                    for i in range(len(dst_tmask_arr)):
                                        if dst_tmask_arr[i] == 1:
                                            merged_rd_data[i] = rd_data[i]
                                    trace['rd_data'] = merged_rd_data
                                else:
                                    trace['rd_data'] = rd_data
                            instr_data[uuid] = trace
                            eop = re.search(eop_pattern, line).group(1) == "1"
                            if eop:
                                tmask_arr = bin_to_array(trace["tmask"])
                                destination = ''
                                if wb:
                                    destination, sep = append_value(destination, trace["rd"], trace['rd_data'], tmask_arr, False)
                                    del trace['rd_data']
                                trace["destination"] = destination
                                operands = ''
                                sep = False
                                if opds[1]:
                                    operands, sep = append_value(operands, trace["rs1"], trace["rs1_data"], tmask_arr, sep)
                                    del trace["rs1_data"]
                                if opds[2]:
                                    operands, sep = append_value(operands, trace["rs2"], trace["rs2_data"], tmask_arr, sep)
                                    del trace["rs2_data"]
                                if opds[3]:
                                    operands, sep = append_value(operands, trace["rs3"], trace["rs3_data"], tmask_arr, sep)
                                    del trace["rs3_data"]
                                trace["operands"] = operands
                                del trace["opds"]
                                del trace["rd"]
                                del trace["rs1"]
                                del trace["rs2"]
                                del trace["rs3"]
                                del trace["issued"]
                                del instr_data[uuid]
                                entries.append(trace)
        except Exception as e:
            print("Error at line {}: {}".format(lineno, e))
    return entries

def write_csv(sublogs, csv_filename, log_type):
    with open(csv_filename, 'w', newline='') as csv_file:
        fieldnames = ["uuid", "PC", "opcode", "instr", "core_id", "warp_id", "tmask", "destination", "operands"]
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

    return sublogs

def main():
    global configs
    args = parse_args()
    configs = load_config(args.log)
    sublogs = split_log_file(args.log)
    write_csv(sublogs, args.csv, args.type)

if __name__ == "__main__":
    main()
