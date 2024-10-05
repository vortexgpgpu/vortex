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

import argparse
import os

def parse_binfile_option(option):
    addr, path = option.split(':')
    return int(addr, 0), path

def parse_value_option(option):
    addr, value = option.split(':')
    return int(addr, 0), value

def load_binary_data(addr, path, word_size, memory, little_endian):
    with open(path, 'rb') as f:
        binary_data = f.read()

    word_count = len(binary_data) // word_size
    if len(binary_data) % word_size != 0:
        word_count += 1

    for i in range(word_count):
        word_data = binary_data[i * word_size: (i + 1) * word_size]
        if little_endian:
            word_data = word_data[::-1]  # Reverse the byte order for little-endian
        hex_value = word_data.hex().zfill(word_size * 2)
        memory[addr + i] = hex_value

def add_value_data(addr, value, memory, word_size):
    value = value.zfill(word_size * 2)
    memory[addr] = value

def binary_to_coe(output_file, word_size, depth, default_value, memory):
    if depth == 0:
        depth = max(memory.keys()) + 1

    with open(output_file, 'w') as coe_file:
        coe_file.write("; This file was generated from binary blobs and/or values\n")
        coe_file.write("memory_initialization_radix=16;\n")
        coe_file.write("memory_initialization_vector=\n")

        for addr in range(depth):
            hex_value = memory.get(addr, default_value)
            coe_file.write(f"{hex_value},\n")

        coe_file.seek(coe_file.tell() - 2)
        coe_file.write(";\n")

def main():
    parser = argparse.ArgumentParser(description="Convert binaries and values to a Xilinx COE file.")
    parser.add_argument("--binfile", action='append', help="Binary file with starting address in the format <addr>:<path>")
    parser.add_argument("--value", action='append', help="Hex value with starting address in the format <addr>:<value>")
    parser.add_argument("--out", default="output.coe", help="Output file (optional).")
    parser.add_argument("--wordsize", type=int, default=4, help="Word size in bytes (default 4).")
    parser.add_argument("--depth", type=int, default=0, help="Address size (optional).")
    parser.add_argument("--default", default="00", help="Default hex value as string (optional).")
    parser.add_argument("--little_endian", action='store_true', help="Interpret binary files as little-endian (default is big-endian).")

    args = parser.parse_args()

    if args.binfile is None and args.value is None:
        raise ValueError("At least one --binfile or --value must be provided.")

    # Initialize memory dictionary
    memory = {}

    # Process binary files
    if args.binfile:
        for option in args.binfile:
            addr, path = parse_binfile_option(option)
            load_binary_data(addr, path, args.wordsize, memory, args.little_endian)

    # Process individual values
    if args.value:
        for option in args.value:
            addr, value = parse_value_option(option)
            add_value_data(addr, value, memory, args.wordsize)

    # Generate the COE file
    binary_to_coe(args.out, args.wordsize, args.depth, args.default.zfill(args.wordsize * 2), memory)

if __name__ == "__main__":
    main()
