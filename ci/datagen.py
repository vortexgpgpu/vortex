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

import struct
import random
import sys

def create_binary_file(n, filename):
    # Open the file in binary write mode
    with open(filename, 'wb') as f:
        # Write the integer N as 4 bytes
        f.write(struct.pack('i', n))
        # Generate and write N floating-point numbers
        for _ in range(n):
            # Generate a random float between 0 and 1
            num = random.random()
            # Write the float in IEEE 754 format (4 bytes)
            f.write(struct.pack('f', num))

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: script.py N filename")
        sys.exit(1)

    n = int(sys.argv[1])
    filename = sys.argv[2]

    create_binary_file(n, filename)
    print(f"Created binary file '{filename}' containing {n} floats.")
