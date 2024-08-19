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

g_memory = {}

def hex2bin(ch):
    return int(ch, 16) if ch.isdigit() or ch in 'abcdefABCDEF' else 0

def process_binary(binfname, wordsize, binaddr):
    with open(binfname, 'rb') as f:
        buffer = list(f.read())
    g_memory[binaddr] = buffer
    return (len(buffer) + wordsize - 1) // wordsize

def process_data(datfname, wordsize):
    offset, buffer = 0, []
    with open(datfname, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith("#"):
                continue
            if line.startswith("@"):
                if buffer:
                    g_memory[offset] = buffer
                offset = int(line[1:], 16)
                buffer = []
            else:
                for i in range(0, len(line), 2):
                    byte = hex2bin(line[i]) << 4 | hex2bin(line[i+1])
                    buffer.append(byte)
                if len(buffer) % wordsize:
                    buffer.extend([0] * (wordsize - len(buffer) % wordsize))
                offset += 1
        if buffer:
            g_memory[offset] = buffer
    return offset

def write_coe(outfname, wordsize, depth, defval):
    with open(outfname, 'w') as f:
        f.write("MEMORY_INITIALIZATION_RADIX=16;\nMEMORY_INITIALIZATION_VECTOR=\n")
        i = 0
        for addr in sorted(g_memory):
            while i < addr:
                f.write(f"{defval},\n")
                i += 1
            data = g_memory[addr]
            for j in range(0, len(data), wordsize):
                f.write(",".join([f"{byte:02x}" for byte in data[j:j+wordsize][::-1]]) + ",\n")
                i += 1
        while i < depth:
            f.write(f"{defval},\n")
            i += 1
        f.seek(f.tell() - 2, 0)  # Remove the last comma
        f.write(";\n")

def main():
    parser = argparse.ArgumentParser(description="Binary to Xilinx COE File Converter")
    parser.add_argument("--binary", help="Input binary file.")
    parser.add_argument("--data", help="Input data file.")
    parser.add_argument("--out", default="output.coe", help="Output file (optional).")
    parser.add_argument("--wordsize", type=int, default=4, help="Word size in bytes (default 4).")
    parser.add_argument("--depth", type=int, default=0, help="Address size (optional).")
    parser.add_argument("--binaddr", type=int, default=0, help="Binary address (optional).")
    parser.add_argument("--default", default="00", help="Default hex value as string (optional).")

    args = parser.parse_args()

    depth = max(
        process_binary(args.binary, args.wordsize, args.binaddr) if args.binary else 0,
        process_data(args.data, args.wordsize) if args.data else 0,
        args.depth
    )

    write_coe(args.out, args.wordsize, depth, args.default)

if __name__ == "__main__":
    main()
