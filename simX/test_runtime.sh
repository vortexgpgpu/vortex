#!/bin/bash

set -e

make
make -C ../tests/runtime/dev
make -C ../tests/runtime/hello
make -C ../tests/runtime/nlTest
make -C ../tests/runtime/simple

./simX -a rv32i -i ../tests/runtime/dev/vx_dev_main.hex
./simX -a rv32i -i ../tests/runtime/hello/hello.hex
./simX -a rv32i -i ../tests/runtime/nlTest/vx_nl_main.hex
./simX -a rv32i -i ../tests/runtime/simple/vx_simple.hex
