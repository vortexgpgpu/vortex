#!/bin/bash

# exit when any command fails
set -e

make -C benchmarks/riscv_tests/isa run
