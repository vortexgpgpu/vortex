 /opt/riscv/bin/riscv32-unknown-linux-gnu-gcc -march=rv32i -mabi=ilp32 -O0 -Wl,-Bstatic,-T,linker.ld -ffreestanding -nostdlib gpgpu_test.c ./lib/lib.c ./lib/queue.c -o gpgpu_test.elf
 /opt/riscv/bin/riscv32-unknown-linux-gnu-objdump -D gpgpu_test.elf > gpgpu_test.dump
 /opt/riscv/bin/riscv32-unknown-linux-gnu-objcopy -O ihex gpgpu_test.elf gpgpu_test.hex