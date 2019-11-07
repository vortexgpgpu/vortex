echo start > results.txt

echo ../kernel/vortex_test.hex
./harptool -E -a rv32i --core ../runtime/vortex_runtime.hex  -s -b 1> emulator.debug
