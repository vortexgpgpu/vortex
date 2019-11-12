echo start > results.txt

# echo ../kernel/vortex_test.hex
./harptool -E -a rv32i --core ../runtime/mains/nlTest/vx_nl_main.hex  -s -b 1> emulator.debug
