#!/bin/bash

set -e

make

echo ../benchmarks/riscv_tests/isa/rv32uf-p-fadd.hex
./simX -a rv32i -r -i ../benchmarks/riscv_tests/isa/rv32uf-p-fadd.hex

echo ../benchmarks/riscv_tests/isa/rv32uf-p-fmadd.hex
./simX -a rv32i -r -i ../benchmarks/riscv_tests/isa/rv32uf-p-fmadd.hex	

echo ../benchmarks/riscv_tests/isa/rv32uf-p-fmin.hex
./simX -a rv32i -r -i ../benchmarks/riscv_tests/isa/rv32uf-p-fmin.hex

echo ../benchmarks/riscv_tests/isa/rv32uf-p-fcmp.hex
./simX -a rv32i -r -i ../benchmarks/riscv_tests/isa/rv32uf-p-fcmp.hex	

echo ../benchmarks/riscv_tests/isa/rv32uf-p-fdst.hex
./simX -a rv32i -r -i ../benchmarks/riscv_tests/isa/rv32uf-p-ldst.hex 

echo ../benchmarks/riscv_tests/isa/rv32uf-p-fcvt.hex
./simX -a rv32i -r -i ../benchmarks/riscv_tests/isa/rv32uf-p-fcvt.hex

echo ../benchmarks/riscv_tests/isa/rv32uf-p-fcvt_w.hex
./simX -a rv32i -r -i ../benchmarks/riscv_tests/isa/rv32uf-p-fcvt_w.hex

echo ../benchmarks/riscv_tests/isa/rv32uf-p-move.hex
./simX -a rv32i -r -i ../benchmarks/riscv_tests/isa/rv32uf-p-move.hex

echo ../benchmarks/riscv_tests/isa/rv32uf-p-recording.hex
./simX -a rv32i -r -i ../benchmarks/riscv_tests/isa/rv32uf-p-recoding.hex

echo ../benchmarks/riscv_tests/isa/rv32uf-p-fdiv.hex
./simX -a rv32i -r -i ../benchmarks/riscv_tests/isa/rv32uf-p-fdiv.hex

echo ../benchmarks/riscv_tests/isa/rv32uf-p-fclass.hex
./simX -a rv32i -r -i ../benchmarks/riscv_tests/isa/rv32uf-p-fclass.hex