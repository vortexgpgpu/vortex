#!/bin/bash

echo start > results.txt

make
printf "Fasten your seatbelts ladies and gentelmen!!\n\n\n\n"
#./simX -a rv32i -i ../benchmarks/vector/vecadd/vx_vec_vecadd.hex  -s 1> emulator.debug
#./simX -a rv32i -i ../benchmarks/vector/saxpy/vx_vec_saxpy.hex  -s 1> emulator.debug
./simX -a rv32i -i ../benchmarks/vector/sgemm_nn/vx_vec_sgemm_nn.hex  -s 1> emulator.debug
