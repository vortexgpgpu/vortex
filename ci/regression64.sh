#!/bin/bash

# exit when any command fails
set -e

# ensure build
XLEN=64 make -s

coverage() 
{
echo "begin coverage tests..."

make -C tests/riscv/isa run-simx-64

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