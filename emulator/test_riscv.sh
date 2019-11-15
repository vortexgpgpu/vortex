echo start > results.txt

# echo ../kernel/vortex_test.hex
./harptool -E -a rv32i --core ../runtime/mains/vecadd/vx_pocl_main.hex  -s -b 1> emulator.debug
