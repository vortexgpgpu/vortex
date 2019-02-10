 /opt/riscv-nommu/bin/riscv32-unknown-linux-gnu-gcc -march=rv32i -mabi=ilp32 -Wl,-Bstatic,-T,linker.ld -ffreestanding -nostdlib firmware.c -o firmware.elf
 /opt/riscv-nommu/bin/riscv32-unknown-linux-gnu-objdump -D firmware.elf > firmware.dump
 /opt/riscv-nommu/bin/riscv32-unknown-linux-gnu-objcopy -O ihex firmware.elf firmware.hex