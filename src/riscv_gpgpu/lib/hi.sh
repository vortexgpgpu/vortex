 /opt/riscv-nommu/bin/riscv32-unknown-linux-gnu-gcc -march=rv32i -mabi=ilp32 -O0 -Wl,-Bstatic,-T,linker.ld -ffreestanding -nostdlib queue.c -o queue.elf
 /opt/riscv-nommu/bin/riscv32-unknown-linux-gnu-objdump -D queue.elf > queue.dump
 /opt/riscv-nommu/bin/riscv32-unknown-linux-gnu-objcopy -O ihex queue.elf queue.hex