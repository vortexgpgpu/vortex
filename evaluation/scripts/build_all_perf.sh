#!/bin/bash

for ((i=1; i <= 16; i=i*2)); do
  echo "Building ${i} core build..."
  ./build.sh -c ${i} -p y
  echo "Done ${i} core build."
done
