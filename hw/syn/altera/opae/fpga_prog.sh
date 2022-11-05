#!/bin/bash

# FPGA programming
# first argument is the bitstream

fpgaconf --bus 0xaf $1
