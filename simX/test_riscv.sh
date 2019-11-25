echo start > results.txt

# echo ../kernel/vortex_test.hex
make
printf "Fasten your seatbelts ladies and gentelmen!!\n\n\n\n"
#cd obj_dir && ./Vcache_simX -E -a rv32i --core ../../runtime/mains/simple/vx_simple_main.hex  -s -b 1> emulator.debug
cd obj_dir && ./Vcache_simX -E -a rv32i --core /home/priya/Desktop/new_vortex/Vortex/rvvector/benchmark_temp/vx_vec_benchmark.hex -s -b 1> emulator.debug
