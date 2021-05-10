#!/bin/bash
# set -e -o pipefail

printf 'algo\top_type\taccel_mode\tinstrs\tcycles\tIPC\n' >results.tsv

aes256_op_types='0 1 2 3 4 5'
# Use some dummy value because the sha test harness does not take any
# "op" argument
sha256_op_types='hash'

for algo in sha256 aes256; do
    pushd "$algo"

    for accel_mode in NATIVE HYBRID SOFTWARE; do
        make clean-all
        CRYPTO_ACCEL_MODE=$accel_mode make

        op_types=${algo}_op_types
        pids=()
        for op_type in ${!op_types}; do
            CRYPTO_OP_TYPE=$op_type make run-fpga &> "run-${op_type}.log"
            pids+=($!)
        done

        # wait "${pids[@]}"

        for op_type in ${!op_types}; do
            {
                printf '%s\t%s\t%s\t' "$algo" "$op_type" "$accel_mode"
                grep PERF "run-${op_type}.log" | sed -e 's/, /\t/g' -e 's/[^0-9.\t]//g'
            } >>../results.tsv
        done
    done

    popd
done
