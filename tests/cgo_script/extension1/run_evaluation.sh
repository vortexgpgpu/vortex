#!/bin/bash 

cd ../../../build 
pwd
./ci/blackbox.sh --cores=4 --warps=16 --threads=32 --l2cache --app=opencl/vecadd

cd tests/opencl
pwd

cp ../../../tests/cgo_script/extension1/test.sh .
cp ../../../tests/cgo_script/extension1/subtest.sh .
cp ../../../tests/cgo_script/extension1/parser.py .

./test.sh

python parser.py
mv output.csv ../../../tests/cgo_script/extension1/output.csv