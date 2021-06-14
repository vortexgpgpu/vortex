#!/bin/bash

set -e

make

echo ./../tests/riscv/isa/rv32ui-p-add.hex
./simX -a rv32i -r -i ../tests/riscv/isa/rv32ui-p-add.hex

echo ./../tests/riscv/isa/rv32ui-p-addi.hex
./simX -a rv32i -r -i ../tests/riscv/isa/rv32ui-p-addi.hex

echo ./../tests/riscv/isa/rv32ui-p-and.hex
./simX -a rv32i -r -i ../tests/riscv/isa/rv32ui-p-and.hex

echo ./../tests/riscv/isa/rv32ui-p-andi.hex
./simX -a rv32i -r -i ../tests/riscv/isa/rv32ui-p-andi.hex

echo ./../tests/riscv/isa/rv32ui-p-auipc.hex
./simX -a rv32i -r -i ../tests/riscv/isa/rv32ui-p-auipc.hex

echo ./../tests/riscv/isa/rv32ui-p-beq.hex
./simX -a rv32i -r -i ../tests/riscv/isa/rv32ui-p-beq.hex

echo ./../tests/riscv/isa/rv32ui-p-bge.hex
./simX -a rv32i -r -i ../tests/riscv/isa/rv32ui-p-bge.hex

echo ./../tests/riscv/isa/rv32ui-p-bgeu.hex
./simX -a rv32i -r -i ../tests/riscv/isa/rv32ui-p-bgeu.hex

echo ./../tests/riscv/isa/rv32ui-p-blt.hex
./simX -a rv32i -r -i ../tests/riscv/isa/rv32ui-p-blt.hex

echo ./../tests/riscv/isa/rv32ui-p-bltu.hex
./simX -a rv32i -r -i ../tests/riscv/isa/rv32ui-p-bltu.hex

echo ./../tests/riscv/isa/rv32ui-p-bne.hex
./simX -a rv32i -r -i ../tests/riscv/isa/rv32ui-p-bne.hex

echo ./../tests/riscv/isa/rv32ui-p-jal.hex
./simX -a rv32i -r -i ../tests/riscv/isa/rv32ui-p-jal.hex

echo ./../tests/riscv/isa/rv32ui-p-jalr.hex
./simX -a rv32i -r -i ../tests/riscv/isa/rv32ui-p-jalr.hex

echo ./../tests/riscv/isa/rv32ui-p-lb.hex
./simX -a rv32i -r -i ../tests/riscv/isa/rv32ui-p-lb.hex

echo ./../tests/riscv/isa/rv32ui-p-lbu.hex
./simX -a rv32i -r -i ../tests/riscv/isa/rv32ui-p-lbu.hex

echo ./../tests/riscv/isa/rv32ui-p-lh.hex
./simX -a rv32i -r -i ../tests/riscv/isa/rv32ui-p-lh.hex

echo ./../tests/riscv/isa/rv32ui-p-lhu.hex
./simX -a rv32i -r -i ../tests/riscv/isa/rv32ui-p-lhu.hex

echo ./../tests/riscv/isa/rv32ui-p-lui.hex
./simX -a rv32i -r -i ../tests/riscv/isa/rv32ui-p-lui.hex

echo ./../tests/riscv/isa/rv32ui-p-lw.hex
./simX -a rv32i -r -i ../tests/riscv/isa/rv32ui-p-lw.hex

echo ./../tests/riscv/isa/rv32ui-p-or.hex
./simX -a rv32i -r -i ../tests/riscv/isa/rv32ui-p-or.hex

echo ./../tests/riscv/isa/rv32ui-p-ori.hex
./simX -a rv32i -r -i ../tests/riscv/isa/rv32ui-p-ori.hex

echo ./../tests/riscv/isa/rv32ui-p-sb.hex
./simX -a rv32i -r -i ../tests/riscv/isa/rv32ui-p-sb.hex

echo ./../tests/riscv/isa/rv32ui-p-sh.hex
./simX -a rv32i -r -i ../tests/riscv/isa/rv32ui-p-sh.hex

echo ./../tests/riscv/isa/rv32ui-p-simple.hex
./simX -a rv32i -r -i ../tests/riscv/isa/rv32ui-p-simple.hex

echo ./../tests/riscv/isa/rv32ui-p-sll.hex
./simX -a rv32i -r -i ../tests/riscv/isa/rv32ui-p-sll.hex

echo ./../tests/riscv/isa/rv32ui-p-slli.hex
./simX -a rv32i -r -i ../tests/riscv/isa/rv32ui-p-slli.hex

echo ./../tests/riscv/isa/rv32ui-p-slt.hex
./simX -a rv32i -r -i ../tests/riscv/isa/rv32ui-p-slt.hex

echo ./../tests/riscv/isa/rv32ui-p-slti.hex
./simX -a rv32i -r -i ../tests/riscv/isa/rv32ui-p-slti.hex

echo ./../tests/riscv/isa/rv32ui-p-sltiu.hex
./simX -a rv32i -r -i ../tests/riscv/isa/rv32ui-p-sltiu.hex

echo ./../tests/riscv/isa/rv32ui-p-sltu.hex
./simX -a rv32i -r -i ../tests/riscv/isa/rv32ui-p-sltu.hex

echo ./../tests/riscv/isa/rv32ui-p-sra.hex
./simX -a rv32i -r -i ../tests/riscv/isa/rv32ui-p-sra.hex

echo ./../tests/riscv/isa/rv32ui-p-srai.hex
./simX -a rv32i -r -i ../tests/riscv/isa/rv32ui-p-srai.hex

echo ./../tests/riscv/isa/rv32ui-p-srl.hex
./simX -a rv32i -r -i ../tests/riscv/isa/rv32ui-p-srl.hex

echo ./../tests/riscv/isa/rv32ui-p-srli.hex
./simX -a rv32i -r -i ../tests/riscv/isa/rv32ui-p-srli.hex

echo ./../tests/riscv/isa/rv32ui-p-sub.hex
./simX -a rv32i -r -i ../tests/riscv/isa/rv32ui-p-sub.hex

echo ./../tests/riscv/isa/rv32ui-p-sw.hex
./simX -a rv32i -r -i ../tests/riscv/isa/rv32ui-p-sw.hex

echo ./../tests/riscv/isa/rv32ui-p-xor.hex
./simX -a rv32i -r -i ../tests/riscv/isa/rv32ui-p-xor.hex

echo ./../tests/riscv/isa/rv32ui-p-xori.hex
./simX -a rv32i -r -i ../tests/riscv/isa/rv32ui-p-xori.hex

echo ./../tests/riscv/isa/rv32um-p-div.hex
./simX -a rv32i -r -i ../tests/riscv/isa/rv32um-p-div.hex

echo ./../tests/riscv/isa/rv32um-p-divu.hex
./simX -a rv32i -r -i ../tests/riscv/isa/rv32um-p-divu.hex

echo ./../tests/riscv/isa/rv32um-p-mul.hex
./simX -a rv32i -r -i ../tests/riscv/isa/rv32um-p-mul.hex

echo ./../tests/riscv/isa/rv32um-p-mulh.hex
./simX -a rv32i -r -i ../tests/riscv/isa/rv32um-p-mulh.hex

echo ./../tests/riscv/isa/rv32um-p-mulhsu.hex
./simX -a rv32i -r -i ../tests/riscv/isa/rv32um-p-mulhsu.hex

echo ./../tests/riscv/isa/rv32um-p-mulhu.hex
./simX -a rv32i -r -i ../tests/riscv/isa/rv32um-p-mulhu.hex

echo ./../tests/riscv/isa/rv32um-p-rem.hex
./simX -a rv32i -r -i ../tests/riscv/isa/rv32um-p-rem.hex

echo ./../tests/riscv/isa/rv32um-p-remu.hex
./simX -a rv32i -r -i ../tests/riscv/isa/rv32um-p-remu.hex