echo start > results.txt

make
printf "Fasten your seatbelts ladies and gentelmen!!\n\n\n\n"
#cd obj_dir && ./Vcache_simX -E -a rv32i --core ../../benchmarks/vector/vecadd/vx_vec_vecadd.hex  -s -b 1> emulator.debug
#cd obj_dir && ./Vcache_simX -E -a rv32i --core ../../benchmarks/vector/saxpy/vx_vec_saxpy.hex  -s -b 1> emulator.debug
cd obj_dir && ./Vcache_simX -E -a rv32i --core ../../benchmarks/vector/sgemm_nn/vx_vec_sgemm_nn.hex  -s -b 1> emulator.debug
