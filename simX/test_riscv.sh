#!/bin/bash

make
echo start > results.txt

echo ./../benchmarks/isa/riscv_tests/rv32ui-p-add.hex >> results.txt
./simX -a rv32i -i ../benchmarks/isa/riscv_tests/rv32ui-p-add.hex -s >> results.txt

echo ./../benchmarks/isa/riscv_tests/rv32ui-p-addi.hex >> results.txt
./simX -a rv32i -i ../benchmarks/isa/riscv_tests/rv32ui-p-addi.hex  -s >> results.txt

echo ./../benchmarks/isa/riscv_tests/rv32ui-p-and.hex >> results.txt
./simX -a rv32i -i ../benchmarks/isa/riscv_tests/rv32ui-p-and.hex  -s >> results.txt

echo ./../benchmarks/isa/riscv_tests/rv32ui-p-andi.hex >> results.txt
./simX -a rv32i -i ../benchmarks/isa/riscv_tests/rv32ui-p-andi.hex  -s >> results.txt

echo ./../benchmarks/isa/riscv_tests/rv32ui-p-auipc.hex >> results.txt
./simX -a rv32i -i ../benchmarks/isa/riscv_tests/rv32ui-p-auipc.hex  -s >> results.txt

echo ./../benchmarks/isa/riscv_tests/rv32ui-p-beq.hex >> results.txt
./simX -a rv32i -i ../benchmarks/isa/riscv_tests/rv32ui-p-beq.hex  -s >> results.txt

echo ./../benchmarks/isa/riscv_tests/rv32ui-p-bge.hex >> results.txt
./simX -a rv32i -i ../benchmarks/isa/riscv_tests/rv32ui-p-bge.hex  -s >> results.txt

echo ./../benchmarks/isa/riscv_tests/rv32ui-p-bgeu.hex >> results.txt
./simX -a rv32i -i ../benchmarks/isa/riscv_tests/rv32ui-p-bgeu.hex  -s >> results.txt

echo ./../benchmarks/isa/riscv_tests/rv32ui-p-blt.hex >> results.txt
./simX -a rv32i -i ../benchmarks/isa/riscv_tests/rv32ui-p-blt.hex  -s >> results.txt

echo ./../benchmarks/isa/riscv_tests/rv32ui-p-bltu.hex >> results.txt
./simX -a rv32i -i ../benchmarks/isa/riscv_tests/rv32ui-p-bltu.hex  -s >> results.txt

echo ./../benchmarks/isa/riscv_tests/rv32ui-p-bne.hex >> results.txt
./simX -a rv32i -i ../benchmarks/isa/riscv_tests/rv32ui-p-bne.hex  -s >> results.txt

echo ./../benchmarks/isa/riscv_tests/rv32ui-p-jal.hex >> results.txt
./simX -a rv32i -i ../benchmarks/isa/riscv_tests/rv32ui-p-jal.hex  -s >> results.txt

echo ./../benchmarks/isa/riscv_tests/rv32ui-p-jalr.hex >> results.txt
./simX -a rv32i -i ../benchmarks/isa/riscv_tests/rv32ui-p-jalr.hex  -s >> results.txt

echo ./../benchmarks/isa/riscv_tests/rv32ui-p-lb.hex >> results.txt
./simX -a rv32i -i ../benchmarks/isa/riscv_tests/rv32ui-p-lb.hex  -s >> results.txt

echo ./../benchmarks/isa/riscv_tests/rv32ui-p-lbu.hex >> results.txt
./simX -a rv32i -i ../benchmarks/isa/riscv_tests/rv32ui-p-lbu.hex  -s >> results.txt

echo ./../benchmarks/isa/riscv_tests/rv32ui-p-lh.hex >> results.txt
./simX -a rv32i -i ../benchmarks/isa/riscv_tests/rv32ui-p-lh.hex  -s >> results.txt

echo ./../benchmarks/isa/riscv_tests/rv32ui-p-lhu.hex >> results.txt
./simX -a rv32i -i ../benchmarks/isa/riscv_tests/rv32ui-p-lhu.hex  -s >> results.txt

echo ./../benchmarks/isa/riscv_tests/rv32ui-p-lui.hex >> results.txt
./simX -a rv32i -i ../benchmarks/isa/riscv_tests/rv32ui-p-lui.hex  -s >> results.txt

echo ./../benchmarks/isa/riscv_tests/rv32ui-p-lw.hex >> results.txt
./simX -a rv32i -i ../benchmarks/isa/riscv_tests/rv32ui-p-lw.hex  -s >> results.txt

echo ./../benchmarks/isa/riscv_tests/rv32ui-p-or.hex >> results.txt
./simX -a rv32i -i ../benchmarks/isa/riscv_tests/rv32ui-p-or.hex  -s >> results.txt

echo ./../benchmarks/isa/riscv_tests/rv32ui-p-ori.hex >> results.txt
./simX -a rv32i -i ../benchmarks/isa/riscv_tests/rv32ui-p-ori.hex  -s >> results.txt

echo ./../benchmarks/isa/riscv_tests/rv32ui-p-sb.hex >> results.txt
./simX -a rv32i -i ../benchmarks/isa/riscv_tests/rv32ui-p-sb.hex  -s >> results.txt

echo ./../benchmarks/isa/riscv_tests/rv32ui-p-sh.hex >> results.txt
./simX -a rv32i -i ../benchmarks/isa/riscv_tests/rv32ui-p-sh.hex  -s >> results.txt

echo ./../benchmarks/isa/riscv_tests/rv32ui-p-simple.hex >> results.txt
./simX -a rv32i -i ../benchmarks/isa/riscv_tests/rv32ui-p-simple.hex  -s >> results.txt

echo ./../benchmarks/isa/riscv_tests/rv32ui-p-sll.hex >> results.txt
./simX -a rv32i -i ../benchmarks/isa/riscv_tests/rv32ui-p-sll.hex  -s >> results.txt

echo ./../benchmarks/isa/riscv_tests/rv32ui-p-slli.hex >> results.txt
./simX -a rv32i -i ../benchmarks/isa/riscv_tests/rv32ui-p-slli.hex  -s >> results.txt

echo ./../benchmarks/isa/riscv_tests/rv32ui-p-slt.hex >> results.txt
./simX -a rv32i -i ../benchmarks/isa/riscv_tests/rv32ui-p-slt.hex  -s >> results.txt

echo ./../benchmarks/isa/riscv_tests/rv32ui-p-slti.hex >> results.txt
./simX -a rv32i -i ../benchmarks/isa/riscv_tests/rv32ui-p-slti.hex  -s >> results.txt

echo ./../benchmarks/isa/riscv_tests/rv32ui-p-sltiu.hex >> results.txt
./simX -a rv32i -i ../benchmarks/isa/riscv_tests/rv32ui-p-sltiu.hex  -s >> results.txt

echo ./../benchmarks/isa/riscv_tests/rv32ui-p-sltu.hex >> results.txt
./simX -a rv32i -i ../benchmarks/isa/riscv_tests/rv32ui-p-sltu.hex  -s >> results.txt

echo ./../benchmarks/isa/riscv_tests/rv32ui-p-sra.hex >> results.txt
./simX -a rv32i -i ../benchmarks/isa/riscv_tests/rv32ui-p-sra.hex  -s >> results.txt

echo ./../benchmarks/isa/riscv_tests/rv32ui-p-srai.hex >> results.txt
./simX -a rv32i -i ../benchmarks/isa/riscv_tests/rv32ui-p-srai.hex  -s >> results.txt

echo ./../benchmarks/isa/riscv_tests/rv32ui-p-srl.hex >> results.txt
./simX -a rv32i -i ../benchmarks/isa/riscv_tests/rv32ui-p-srl.hex  -s >> results.txt

echo ./../benchmarks/isa/riscv_tests/rv32ui-p-srli.hex >> results.txt
./simX -a rv32i -i ../benchmarks/isa/riscv_tests/rv32ui-p-srli.hex  -s >> results.txt

echo ./../benchmarks/isa/riscv_tests/rv32ui-p-sub.hex >> results.txt
./simX -a rv32i -i ../benchmarks/isa/riscv_tests/rv32ui-p-sub.hex  -s >> results.txt

echo ./../benchmarks/isa/riscv_tests/rv32ui-p-sw.hex >> results.txt
./simX -a rv32i -i ../benchmarks/isa/riscv_tests/rv32ui-p-sw.hex  -s >> results.txt

echo ./../benchmarks/isa/riscv_tests/rv32ui-p-xor.hex >> results.txt
./simX -a rv32i -i ../benchmarks/isa/riscv_tests/rv32ui-p-xor.hex  -s >> results.txt

echo ./../benchmarks/isa/riscv_tests/rv32ui-p-xori.hex >> results.txt
./simX -a rv32i -i ../benchmarks/isa/riscv_tests/rv32ui-p-xori.hex  -s >> results.txt

# echo ./../benchmarks/isa/riscv_tests/rv32um-p-div.hex >> results.txt
# ./simX -a rv32i -i ../benchmarks/isa/riscv_tests/rv32um-p-div.hex  -s >> results.txt

# echo ./../benchmarks/isa/riscv_tests/rv32um-p-divu.hex >> results.txt
# ./simX -a rv32i -i ../benchmarks/isa/riscv_tests/rv32um-p-divu.hex  -s >> results.txt

# echo ./../benchmarks/isa/riscv_tests/rv32um-p-mul.hex >> results.txt
# ./simX -a rv32i -i ../benchmarks/isa/riscv_tests/rv32um-p-mul.hex  -s >> results.txt

# echo ./../benchmarks/isa/riscv_tests/rv32um-p-mulh.hex >> results.txt
# ./simX -a rv32i -i ../benchmarks/isa/riscv_tests/rv32um-p-mulh.hex  -s >> results.txt

# echo ./../benchmarks/isa/riscv_tests/rv32um-p-mulhsu.hex >> results.txt
# ./simX -a rv32i -i ../benchmarks/isa/riscv_tests/rv32um-p-mulhsu.hex  -s >> results.txt

# echo ./../benchmarks/isa/riscv_tests/rv32um-p-mulhu.hex >> results.txt
# ./simX -a rv32i -i ../benchmarks/isa/riscv_tests/rv32um-p-mulhu.hex  -s >> results.txt

# echo ./../benchmarks/isa/riscv_tests/rv32um-p-rem.hex >> results.txt
# ./simX -a rv32i -i ../benchmarks/isa/riscv_tests/rv32um-p-rem.hex  -s >> results.txt

# echo ./../benchmarks/isa/riscv_tests/rv32um-p-remu.hex >> results.txt
# ./simX -a rv32i -i ../benchmarks/isa/riscv_tests/rv32um-p-remu.hex  -s >> results.txt
