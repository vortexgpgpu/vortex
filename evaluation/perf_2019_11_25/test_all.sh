

for PROJECT in sfilter; do
    echo "" > $PROJECT.result
    for number_of_warps in 2 4 8 16 32; do
        for number_of_threads in 2 4 8 16 32; do

        	echo "$PROJECT = Warp Count: $number_of_warps  Thread Count: $number_of_threads Launched"
            echo "#define TOTAL_THREADS $number_of_threads" > ../../runtime/config.h
            echo "#define TOTAL_WARPS $number_of_warps"  >> ../../runtime/config.h

            cd ../opencl/$PROJECT
            make clean &>> /dev/null
            make &>> /dev/null
            cd ../../test_benchmark

            echo "Warps: $number_of_warps, Threads: $number_of_threads" >> $PROJECT.result

            # echo ../../../simX/obj_dir/Vcache_simX -E -a rv32i --core ../opencl/$PROJECT/$PROJECT.hex -s -b &>> $PROJECT.result

            ../../simX/obj_dir/Vcache_simX -E -a rv32i --core ../opencl/$PROJECT/$PROJECT.hex -s -b &>> $PROJECT.result


        done
    done

done