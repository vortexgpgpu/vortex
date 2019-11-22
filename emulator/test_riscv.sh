echo start > results.txt

# echo ../kernel/vortex_test.hex
# ./harptool -E -a rv32i --core ../runtime/mains/simple/vx_simple_main.hex  -s -b 1> emulator.debug
./harptool -E -a rv32i --core ../benchmarks/opencl/sgemm/sgemm.hex  -s -b 1> emulator.debug
# ./harptool -E -a rv32i --core ../runtime/mains/vector_test/vx_vector_main.hex  -s -b 1> emulator.debug
