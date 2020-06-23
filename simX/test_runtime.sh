#!/bin/bash

make
make -C ../runtime/tests/dev
make -C ../runtime/tests/hello
make -C ../runtime/tests/nativevecadd
make -C ../runtime/tests/simple
make -C ../runtime/tests/vecadd

cd obj_dir
echo start > results.txt

printf "Fasten your seatbelts ladies and gentelmen!!\n\n\n\n"

#./Vcache_simX -E -a rv32i --core ../runtime/tests/dev/vx_dev_main.hex  -s -b 1> emulator.debug
#./Vcache_simX -E -a rv32i --core ../runtime/tests/hello/hello.hex  -s -b 1> emulator.debug
./Vcache_simX -E -a rv32i --core ../runtime/tests/nativevecadd/vx_pocl_main.hex  -s -b 1> emulator.debug
./Vcache_simX -E -a rv32i --core ../runtime/tests/simple/vx_simple_main.hex  -s -b 1> emulator.debug
./Vcache_simX -E -a rv32i --core ../runtime/tests/vecadd/vx_pocl_main.hex  -s -b 1> emulator.debug
