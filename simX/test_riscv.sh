make
cd obj_dir
echo start > results.txt

echo ./riscv_tests/rv32ui-p-add.hex >> results.txt
./Vcache_simX -E --cpu -a rv32i --core ../riscv_tests/rv32ui-p-add.hex -s -b >> results.txt

echo ./riscv_tests/rv32ui-p-addi.hex >> results.txt
./Vcache_simX -E --cpu -a rv32i --core ../riscv_tests/rv32ui-p-addi.hex  -s -b >> results.txt

echo ./riscv_tests/rv32ui-p-and.hex >> results.txt
./Vcache_simX -E --cpu -a rv32i --core ../riscv_tests/rv32ui-p-and.hex  -s -b >> results.txt

echo ./riscv_tests/rv32ui-p-andi.hex >> results.txt
./Vcache_simX -E --cpu -a rv32i --core ../riscv_tests/rv32ui-p-andi.hex  -s -b >> results.txt

echo ./riscv_tests/rv32ui-p-auipc.hex >> results.txt
./Vcache_simX -E --cpu -a rv32i --core ../riscv_tests/rv32ui-p-auipc.hex  -s -b >> results.txt

echo ./riscv_tests/rv32ui-p-beq.hex >> results.txt
./Vcache_simX -E --cpu -a rv32i --core ../riscv_tests/rv32ui-p-beq.hex  -s -b >> results.txt

echo ./riscv_tests/rv32ui-p-bge.hex >> results.txt
./Vcache_simX -E --cpu -a rv32i --core ../riscv_tests/rv32ui-p-bge.hex  -s -b >> results.txt

echo ./riscv_tests/rv32ui-p-bgeu.hex >> results.txt
./Vcache_simX -E --cpu -a rv32i --core ../riscv_tests/rv32ui-p-bgeu.hex  -s -b >> results.txt

echo ./riscv_tests/rv32ui-p-blt.hex >> results.txt
./Vcache_simX -E --cpu -a rv32i --core ../riscv_tests/rv32ui-p-blt.hex  -s -b >> results.txt

echo ./riscv_tests/rv32ui-p-bltu.hex >> results.txt
./Vcache_simX -E --cpu -a rv32i --core ../riscv_tests/rv32ui-p-bltu.hex  -s -b >> results.txt

echo ./riscv_tests/rv32ui-p-bne.hex >> results.txt
./Vcache_simX -E --cpu -a rv32i --core ../riscv_tests/rv32ui-p-bne.hex  -s -b >> results.txt

echo ./riscv_tests/rv32ui-p-jal.hex >> results.txt
./Vcache_simX -E --cpu -a rv32i --core ../riscv_tests/rv32ui-p-jal.hex  -s -b >> results.txt

echo ./riscv_tests/rv32ui-p-jalr.hex >> results.txt
./Vcache_simX -E --cpu -a rv32i --core ../riscv_tests/rv32ui-p-jalr.hex  -s -b >> results.txt

echo ./riscv_tests/rv32ui-p-lb.hex >> results.txt
./Vcache_simX -E --cpu -a rv32i --core ../riscv_tests/rv32ui-p-lb.hex  -s -b >> results.txt

echo ./riscv_tests/rv32ui-p-lbu.hex >> results.txt
./Vcache_simX -E --cpu -a rv32i --core ../riscv_tests/rv32ui-p-lbu.hex  -s -b >> results.txt

echo ./riscv_tests/rv32ui-p-lh.hex >> results.txt
./Vcache_simX -E --cpu -a rv32i --core ../riscv_tests/rv32ui-p-lh.hex  -s -b >> results.txt

echo ./riscv_tests/rv32ui-p-lhu.hex >> results.txt
./Vcache_simX -E --cpu -a rv32i --core ../riscv_tests/rv32ui-p-lhu.hex  -s -b >> results.txt

echo ./riscv_tests/rv32ui-p-lui.hex >> results.txt
./Vcache_simX -E --cpu -a rv32i --core ../riscv_tests/rv32ui-p-lui.hex  -s -b >> results.txt

echo ./riscv_tests/rv32ui-p-lw.hex >> results.txt
./Vcache_simX -E --cpu -a rv32i --core ../riscv_tests/rv32ui-p-lw.hex  -s -b >> results.txt

echo ./riscv_tests/rv32ui-p-or.hex >> results.txt
./Vcache_simX -E --cpu -a rv32i --core ../riscv_tests/rv32ui-p-or.hex  -s -b >> results.txt

echo ./riscv_tests/rv32ui-p-ori.hex >> results.txt
./Vcache_simX -E --cpu -a rv32i --core ../riscv_tests/rv32ui-p-ori.hex  -s -b >> results.txt

echo ./riscv_tests/rv32ui-p-sb.hex >> results.txt
./Vcache_simX -E --cpu -a rv32i --core ../riscv_tests/rv32ui-p-sb.hex  -s -b >> results.txt

echo ./riscv_tests/rv32ui-p-sh.hex >> results.txt
./Vcache_simX -E --cpu -a rv32i --core ../riscv_tests/rv32ui-p-sh.hex  -s -b >> results.txt

echo ./riscv_tests/rv32ui-p-simple.hex >> results.txt
./Vcache_simX -E --cpu -a rv32i --core ../riscv_tests/rv32ui-p-simple.hex  -s -b >> results.txt

echo ./riscv_tests/rv32ui-p-sll.hex >> results.txt
./Vcache_simX -E --cpu -a rv32i --core ../riscv_tests/rv32ui-p-sll.hex  -s -b >> results.txt

echo ./riscv_tests/rv32ui-p-slli.hex >> results.txt
./Vcache_simX -E --cpu -a rv32i --core ../riscv_tests/rv32ui-p-slli.hex  -s -b >> results.txt

echo ./riscv_tests/rv32ui-p-slt.hex >> results.txt
./Vcache_simX -E --cpu -a rv32i --core ../riscv_tests/rv32ui-p-slt.hex  -s -b >> results.txt

echo ./riscv_tests/rv32ui-p-slti.hex >> results.txt
./Vcache_simX -E --cpu -a rv32i --core ../riscv_tests/rv32ui-p-slti.hex  -s -b >> results.txt

echo ./riscv_tests/rv32ui-p-sltiu.hex >> results.txt
./Vcache_simX -E --cpu -a rv32i --core ../riscv_tests/rv32ui-p-sltiu.hex  -s -b >> results.txt

echo ./riscv_tests/rv32ui-p-sltu.hex >> results.txt
./Vcache_simX -E --cpu -a rv32i --core ../riscv_tests/rv32ui-p-sltu.hex  -s -b >> results.txt

echo ./riscv_tests/rv32ui-p-sra.hex >> results.txt
./Vcache_simX -E --cpu -a rv32i --core ../riscv_tests/rv32ui-p-sra.hex  -s -b >> results.txt

echo ./riscv_tests/rv32ui-p-srai.hex >> results.txt
./Vcache_simX -E --cpu -a rv32i --core ../riscv_tests/rv32ui-p-srai.hex  -s -b >> results.txt

echo ./riscv_tests/rv32ui-p-srl.hex >> results.txt
./Vcache_simX -E --cpu -a rv32i --core ../riscv_tests/rv32ui-p-srl.hex  -s -b >> results.txt

echo ./riscv_tests/rv32ui-p-srli.hex >> results.txt
./Vcache_simX -E --cpu -a rv32i --core ../riscv_tests/rv32ui-p-srli.hex  -s -b >> results.txt

echo ./riscv_tests/rv32ui-p-sub.hex >> results.txt
./Vcache_simX -E --cpu -a rv32i --core ../riscv_tests/rv32ui-p-sub.hex  -s -b >> results.txt

echo ./riscv_tests/rv32ui-p-sw.hex >> results.txt
./Vcache_simX -E --cpu -a rv32i --core ../riscv_tests/rv32ui-p-sw.hex  -s -b >> results.txt

echo ./riscv_tests/rv32ui-p-xor.hex >> results.txt
./Vcache_simX -E --cpu -a rv32i --core ../riscv_tests/rv32ui-p-xor.hex  -s -b >> results.txt

echo ./riscv_tests/rv32ui-p-xori.hex >> results.txt
./Vcache_simX -E --cpu -a rv32i --core ../riscv_tests/rv32ui-p-xori.hex  -s -b >> results.txt

# echo ./riscv_tests/rv32um-p-div.hex >> results.txt
# ./Vcache_simX -E --cpu -a rv32i --core ../riscv_tests/rv32um-p-div.hex  -s -b >> results.txt

# echo ./riscv_tests/rv32um-p-divu.hex >> results.txt
# ./Vcache_simX -E --cpu -a rv32i --core ../riscv_tests/rv32um-p-divu.hex  -s -b >> results.txt

# echo ./riscv_tests/rv32um-p-mul.hex >> results.txt
# ./Vcache_simX -E --cpu -a rv32i --core ../riscv_tests/rv32um-p-mul.hex  -s -b >> results.txt

# echo ./riscv_tests/rv32um-p-mulh.hex >> results.txt
# ./Vcache_simX -E --cpu -a rv32i --core ../riscv_tests/rv32um-p-mulh.hex  -s -b >> results.txt

# echo ./riscv_tests/rv32um-p-mulhsu.hex >> results.txt
# ./Vcache_simX -E --cpu -a rv32i --core ../riscv_tests/rv32um-p-mulhsu.hex  -s -b >> results.txt

# echo ./riscv_tests/rv32um-p-mulhu.hex >> results.txt
# ./Vcache_simX -E --cpu -a rv32i --core ../riscv_tests/rv32um-p-mulhu.hex  -s -b >> results.txt

# echo ./riscv_tests/rv32um-p-rem.hex >> results.txt
# ./Vcache_simX -E --cpu -a rv32i --core ../riscv_tests/rv32um-p-rem.hex  -s -b >> results.txt

# echo ./riscv_tests/rv32um-p-remu.hex >> results.txt
# ./Vcache_simX -E --cpu -a rv32i --core ../riscv_tests/rv32um-p-remu.hex  -s -b >> results.txt

