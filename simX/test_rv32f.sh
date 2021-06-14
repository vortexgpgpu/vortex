#!/bin/bash

set -e

make

echo ../tests/riscv/isa/rv32uf-p-fadd.hex
./simX -a rv32i -r -i ../tests/riscv/isa/rv32uf-p-fadd.hex

echo ../tests/riscv/isa/rv32uf-p-fmadd.hex
./simX -a rv32i -r -i ../tests/riscv/isa/rv32uf-p-fmadd.hex	

echo ../tests/riscv/isa/rv32uf-p-fmin.hex
./simX -a rv32i -r -i ../tests/riscv/isa/rv32uf-p-fmin.hex

echo ../tests/riscv/isa/rv32uf-p-fcmp.hex
./simX -a rv32i -r -i ../tests/riscv/isa/rv32uf-p-fcmp.hex	

echo ../tests/riscv/isa/rv32uf-p-fdst.hex
./simX -a rv32i -r -i ../tests/riscv/isa/rv32uf-p-ldst.hex 

echo ../tests/riscv/isa/rv32uf-p-fcvt.hex
./simX -a rv32i -r -i ../tests/riscv/isa/rv32uf-p-fcvt.hex

echo ../tests/riscv/isa/rv32uf-p-fcvt_w.hex
./simX -a rv32i -r -i ../tests/riscv/isa/rv32uf-p-fcvt_w.hex

echo ../tests/riscv/isa/rv32uf-p-move.hex
./simX -a rv32i -r -i ../tests/riscv/isa/rv32uf-p-move.hex

echo ../tests/riscv/isa/rv32uf-p-recording.hex
./simX -a rv32i -r -i ../tests/riscv/isa/rv32uf-p-recoding.hex

echo ../tests/riscv/isa/rv32uf-p-fdiv.hex
./simX -a rv32i -r -i ../tests/riscv/isa/rv32uf-p-fdiv.hex

echo ../tests/riscv/isa/rv32uf-p-fclass.hex
./simX -a rv32i -r -i ../tests/riscv/isa/rv32uf-p-fclass.hex