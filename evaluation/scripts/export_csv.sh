#!/bin/bash

while getopts c:d:i:o:p: flag
do
  case "${flag}" in
    c) cores=${OPTARG};; #1, 2, 4, 8, 16
    d) dir=${OPTARG};; #directory name (e.g. perf_2021_03_07)
    i) ifile=${OPTARG};; #input filename
    o) ofile=${OPTARG};; #output filename
    p) param=${OPTARG};; #parameter to be made into csv
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

printf "cores,${param}\n" > "../${dir}/${ofile}"
for ((i=1; i<=$cores; i=i*2)); do
  printf "${i}," >> "../${dir}/${ofile}"
  (sed -n "s/${param}=\(.*\)/\1/p" < "../${dir}/${i}c/${ifile}") >> "../${dir}/${ofile}"
done
