#!/bin/bash

set -e

make
make -C ../runtime/tests/dev
make -C ../runtime/tests/hello
make -C ../runtime/tests/nlTest
make -C ../runtime/tests/simple

./simX -a rv32i -i ../runtime/tests/dev/vx_dev_main.hex
./simX -a rv32i -i ../runtime/tests/hello/hello.hex
./simX -a rv32i -i ../runtime/tests/nlTest/vx_nl_main.hex
./simX -a rv32i -i ../runtime/tests/simple/vx_simple.hex
