set SOURCE_FILES [glob *.lib]
foreach FILE ${SOURCE_FILES} {
    read_lib $FILE
    redirect -variable CURR_LIB {get_lib}

    set CURR_LIB [string range $CURR_LIB 2 end-3] 
    set CURR_LIB [lindex  $CURR_LIB 0]
    set FILENAME [string range $FILE 0 end-4]
    write_lib $CURR_LIB -output ${FILENAME}.db
    remove_lib $CURR_LIB
}

exit
