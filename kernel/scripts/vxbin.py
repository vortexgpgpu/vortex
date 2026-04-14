#!/usr/bin/env python3

# Copyright 2019-2023
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import subprocess
import struct
import sys
import re

def get_vma_size(elf_file):
    try:
        cmd = ['readelf', '-l', '-W', elf_file]
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)
        output, errors = process.communicate()
        if process.returncode != 0:
            print("Error running readelf: {}".format(errors.strip()))
            sys.exit(-1)

        min_vma = 2**64 - 1
        max_vma = 0
        regex = re.compile(r'\s*LOAD\s+(\w+)\s+(\w+)\s+(\w+)\s+(\w+)\s+(\w+)')

        for line in output.splitlines():
            match = regex.match(line)
            if match:
                vma = int(match.group(2), 16)
                size = int(match.group(5), 16)
                end_vma = vma + size
                min_vma = min(min_vma, vma)
                max_vma = max(max_vma, end_vma)
                vma_size = max_vma - min_vma
                #print("vma={0:x}, size={1}, min_vma=0x{2:x}, max_vma=0x{3:x}, vma_size={4}".format(vma, size, min_vma, max_vma, vma_size))

        return min_vma, max_vma

    except Exception as e:
        print("Failed to calculate vma size due to an error: {}".format(str(e)))
        sys.exit(-1)

def get_symbol(elf_file, name):
    # Read a symbol value from the ELF. We use _edata as the start of BSS and
    # _end as the end of BSS so runtime_size covers the full RW region (the
    # linker's DATA_SEGMENT_ALIGN can push _edata/_end past the end of the last
    # LOAD segment when the kernel has little/no data or BSS).
    cmd = ['readelf', '-s', '-W', elf_file]
    output = subprocess.check_output(cmd, universal_newlines=True)
    regex = re.compile(r'\s*\d+:\s+([0-9a-fA-F]+)\s+\d+\s+\S+\s+\S+\s+\S+\s+\S+\s+' + re.escape(name) + r'$')
    for line in output.splitlines():
        match = regex.match(line)
        if match:
            return int(match.group(1), 16)
    print("Error: {} symbol not found in {}".format(name, elf_file))
    sys.exit(-1)

def create_vxbin_binary(input_elf, output_bin, objcopy_path):
    min_vma, max_vma = get_vma_size(input_elf)
    edata = get_symbol(input_elf, '_edata')
    end = get_symbol(input_elf, '_end')

    # Extend max_vma to _end so runtime_size covers the BSS region even when
    # the linker aligns _edata/_end past the last LOAD segment.
    max_vma = max(max_vma, end)

    # Create a binary data from the ELF file using objcopy. Use a per-call
    # unique tempfile so parallel builds don't race on a shared path.
    import tempfile
    fd, temp_bin_path = tempfile.mkstemp(prefix='vxbin_', suffix='.bin')
    os.close(fd)
    subprocess.check_call([objcopy_path, '-O', 'binary', input_elf, temp_bin_path])

    # Read the binary file to determine its size
    with open(temp_bin_path, 'rb') as temp_file:
        binary_data = temp_file.read()

    # Pad the payload up to _edata so that bin_size reflects the kernel's
    # boundary between data and BSS (and inherits _edata's cache alignment).
    expected_bin_size = edata - min_vma
    if len(binary_data) < expected_bin_size:
        binary_data += b'\x00' * (expected_bin_size - len(binary_data))

    # Pack addresses into 64-bit unsigned integer
    min_vma_bytes = struct.pack('<Q', min_vma)
    max_vma_bytes = struct.pack('<Q', max_vma)

    # Write the total size and binary data to the final output file
    with open(output_bin, 'wb') as bin_file:
        bin_file.write(min_vma_bytes)
        bin_file.write(max_vma_bytes)
        bin_file.write(binary_data)

    # Remove the temporary binary file
    os.remove(temp_bin_path)
    # print("Binary created successfully: {}, min_vma={:x}, max_vma={:x}".format(output_bin, min_vma, max_vma))

if __name__ == '__main__':
    if len(sys.argv) != 3:
        print("Usage: vxbin.py <input>.elf <output>.vxbin")
        sys.exit(-1)

    objcopy_path = os.getenv('OBJCOPY', 'objcopy')  # Default to 'objcopy' if not set

    create_vxbin_binary(sys.argv[1], sys.argv[2], objcopy_path)
