#!/bin/bash

# exit when any command fails
set -e

# ensure build
make -s

coverage() 
{
echo "begin coverage tests..."

make -C sim/simx clean
XLEN=64 make -C sim/simx
XLEN=64 make -C tests/riscv/isa run-simx

echo "coverage tests done!"
}

usage()
{
    echo "usage: regression [-coverage] [-all] [-h|--help]"
}

while [ "$1" != "" ]; do
    case $1 in
        -coverage ) coverage
                ;;
        -all ) coverage
                ;;
        -h | --help ) usage
                      exit
                ;;
        * )           usage
                      exit 1
    esac
    shift
done