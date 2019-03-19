echo start > results.txt

echo ./vortex_software/vortex_test.hex
./harptool -E -a rv32i --core ./vortex_software/vortex_test.hex  -s -b
