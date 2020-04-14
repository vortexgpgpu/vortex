#! /bin/csh

setenv SNPSLMD_LICENSE_FILE 1910@ece-winlic.ece.gatech.edu
setenv PATH "${PATH}:/tools/synopsys/synthesis/j201409sp3/bin"
setenv SYNOPSYS /tools/synopsys/synthesis/j201409sp3

foreach ram (`ls`)
	if ( -d ./$ram ) then
		echo $ram
		cd $ram
		lc_shell -f ../convert_lib_to_db.tcl
		cd ..
	endif
end
