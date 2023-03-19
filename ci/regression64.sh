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

show_usage()
{
    echo "Vortex 64-bit Regression Test"
    echo "Usage: $0 [-coverage] [-all] [-h|--help]"
}

while [ "$1" != "" ]; do
    case $1 in
        -coverage ) coverage
                ;;
        -all ) coverage
                ;;
        -h | --help ) show_usage
                      exit
                ;;
        * )           show_usage
                      exit 1
    esac
    shift
done