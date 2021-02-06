#!/bin/bash

echo start > results.txt

# echo ../kernel/vortex_test.hex
make
printf "Fasten your seatbelts ladies and gentelmen!!\n\n\n\n"
./Vcache_simX -a rv32i -i ../rvvector/basic/vx_vector_main.hex -s 1> emulator.debug
