echo start > results.txt

echo ../kernel/vortex_test.hex
./harptool -E -a rv32i --core ../kernel/vortex_test.hex  -s -b
