 /opt/riscv/bin/riscv32-unknown-elf-gcc -march=rv32i -mabi=ilp32 -O0 -Wl,-Bstatic,-T,linker.ld -ffreestanding -nostdlib ./lib/lib.s gpgpu_test.c ./lib/queue.s ./lib/lib.c -o gpgpu_test.elf
 /opt/riscv/bin/riscv32-unknown-elf-objdump -D gpgpu_test.elf > gpgpu_test.dump
 /opt/riscv/bin/riscv32-unknown-elf-objcopy -O ihex gpgpu_test.elf gpgpu_test.hex
