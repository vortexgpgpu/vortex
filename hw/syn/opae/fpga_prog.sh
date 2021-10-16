#!/bin/bash

# FPGA programming
# first argument is the bitstream

echo "fpgaconf --bus 0xaf $1"
fpgaconf --bus 0xaf $1