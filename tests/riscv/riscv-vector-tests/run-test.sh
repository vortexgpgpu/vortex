#!/bin/bash
cd testcases
rm "$1"*
cp ../../../../third_party/riscv-vector-tests/out/v"$VLEN"x"$XLEN"machine/bin/stage2/"$1"* .
passed=0
failed=0
all=0
for f in "$1"* ; do 
  mv "$f" "$f.elf";
  "$RISCV_TOOLCHAIN_PATH"/bin/riscv"$XLEN"-unknown-elf-objdump -D "$f.elf" > "$f.dump";
  "$RISCV_TOOLCHAIN_PATH"/bin/riscv"$XLEN"-unknown-elf-objcopy -O binary "$f.elf" "$f.bin";
  ../../../../sim/simx/simx -c 1 "$f.bin" &> "$f.log";
  if [ $? -eq 0 ]; then
    echo "$f PASSED"
    let "passed++"
  else
    echo "$f FAILED"
    let "failed++"
  fi
  let "all++"
done
cd ..
echo "Passed $passed out of $all vector tests."
exit $failed