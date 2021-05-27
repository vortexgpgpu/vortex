#!/bin/bash

while getopts c:d:f:o: flag
do
  case "${flag}" in
    c) cores=${OPTARG};; #1, 2, 4, 8, 16
    d) dir=${OPTARG};; #directory name (e.g. perf_2021_03_07)
    i) ifile=${OPTARG};; #input filename
    o) ofile=${OPTARG};; #output filename
  esac
done

if [[ ! "$cores" =~ ^(1|2|4|8|16)$ ]]; then
  echo 'Invalid parameter for argument -c (1, 2, 4, 8, or 16 expected)'
  exit 1
fi

if [ -z "$ifile" ]; then
  echo 'No input filename given for argument -f'
  exit 1
fi

if [ -z "$dir" ]; then
  echo 'No directory given for argument -d'
  exit 1
fi

printf "cores,IPC" > "../${dir}/${ofile}"
for ((i=1; i<=$cores; i=i*2)); do
  printf "${i}," >> "../${dir}/${ofile}"
  (sed -n "s/IPC=\(.*\)/\1/p" < "../${dir}/${i}c/${ifile}" | awk 'END {print $NF}') >> "../${dir}/${ofile}"
done
