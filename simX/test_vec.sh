#!/bin/bash

echo start > results.txt

# echo ../kernel/vortex_test.hex
make
printf "Fasten your seatbelts ladies and gentelmen!!\n\n\n\n"
cd obj_dir && ./Vcache_simX -E -a rv32i --core ../../rvvector/basic/vx_vector_main.hex  -s -b 1> emulator.debug
