#!/bin/bash

make
make -C ../runtime/tests/dev
make -C ../runtime/tests/hello
make -C ../runtime/tests/nlTest
make -C ../runtime/tests/simple

echo start > results.txt

printf "Fasten your seatbelts ladies and gentelmen!!\n\n\n\n"

#./simX -a rv32i -i ../runtime/tests/dev/vx_dev_main.hex -s 1> emulator.debug
#./simX -a rv32i -i ../runtime/tests/hello/hello.hex -s 1> emulator.debug
./simX -a rv32i -i ../runtime/tests/nlTest/vx_nl_main.hex -s 1> emulator.debug
./simX -a rv32i -i ../runtime/tests/simple/vx_simple_main.hex -s 1> emulator.debug
