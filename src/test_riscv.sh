echo start > results.txt

echo ./riscv_gpgpu/gpgpu_test.hex
./harptool -E -a rv32i --core ./riscv_gpgpu/gpgpu_test.hex  -s -b
