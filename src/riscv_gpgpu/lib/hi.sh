/opt/riscv/bin/riscv32-unknown-elf-gcc -march=rv32i -mabi=ilp32 -O0 -Wl,-Bstatic,-T,linker.ld -ffreestanding -nostdlib queue.c -o queue.elf
/opt/riscv/bin/riscv32-unknown-elf-objdump -D queue.elf > queue.dump
/opt/riscv/bin/riscv32-unknown-elf-objcopy -O ihex queue.elf queue.hex